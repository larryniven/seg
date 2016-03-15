#include "scrf/scrf_util.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_weight.h"
#include <cassert>
#include <fstream>

namespace scrf {

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm)
    {
        lm::fst result = *lm;
        result.data = std::make_shared<lm::fst_data>(*(lm->data));
        result.data->in_edges_map.clear();
        result.data->in_edges_map.resize(result.data->edges.size());
        result.data->out_edges_map.clear();
        result.data->out_edges_map.resize(result.data->edges.size());
        for (int e = 0; e < result.data->edges.size(); ++e) {
            auto& e_data = result.data->edges.at(e);
            e_data.input = "<label>";
            int tail = result.tail(e);
            int head = result.head(e);
            result.data->out_edges_map[tail]["<label>"].push_back(e);
            result.data->in_edges_map[head]["<label>"].push_back(e);
        }

        return std::make_shared<lm::fst>(result);
    }

    fst::path<scrf::scrf_t> make_min_cost_path(
        scrf::scrf_t& min_cost,
        fst::path<scrf::scrf_t> const& gold_path)
    {
        min_cost.weight_func =
            std::make_shared<scrf::neg_cost>(
                scrf::neg_cost { std::make_shared<scrf::seg_cost>(
                scrf::make_overlap_cost(gold_path)) })
            + std::make_shared<scrf::backoff_cost>(scrf::backoff_cost{});
        min_cost.topo_order = scrf::topo_order(min_cost);
        fst::path<scrf::scrf_t> min_cost_path = scrf::shortest_path(
            min_cost, min_cost.topo_order);
    
        double min_cost_path_weight = 0;
    
        std::cout << "min cost path: ";
        for (auto& e: min_cost_path.edges()) {
            int tail = std::get<0>(min_cost_path.tail(e));
            int head = std::get<0>(min_cost_path.head(e));
    
            std::cout << min_cost_path.output(e) << " ";
            min_cost_path_weight += min_cost_path.weight(e);
        }
        std::cout << std::endl;
    
        std::cout << "cost: " << min_cost_path_weight << std::endl;
    
        return min_cost_path;
    }
    
    scrf_t make_gold_scrf(lattice::fst gold_lat,
        std::shared_ptr<lm::fst> lm)
    {
        gold_lat.data = std::make_shared<lattice::fst_data>(*(gold_lat.data));
        lattice::add_eps_loops(gold_lat);
        fst::composed_fst<lattice::fst, lm::fst> gold_lm_lat;
        gold_lm_lat.fst1 = std::make_shared<lattice::fst>(std::move(gold_lat));
        gold_lm_lat.fst2 = lm;

        scrf_t gold;
        gold.fst = std::make_shared<decltype(gold_lm_lat)>(gold_lm_lat);

        return gold;
    }

    fst::path<scrf::scrf_t> make_ground_truth_path(
        scrf::scrf_t& ground_truth)
    {
        ground_truth.weight_func = std::make_shared<scrf::backoff_cost>(scrf::backoff_cost{});
        ground_truth.topo_order = scrf::topo_order(ground_truth);
        return scrf::shortest_path(ground_truth, ground_truth.topo_order);
    }

    lattice::fst make_segmentation_lattice(int frames, int min_seg_len, int max_seg_len)
    {
        lattice::fst_data data;

        data.vertices.resize(frames + 1);
        for (int i = 0; i < frames + 1; ++i) {
            data.vertices.at(i).time = i;
        }

        data.in_edges.resize(frames + 1);
        data.out_edges.resize(frames + 1);
        data.in_edges_map.resize(frames + 1);
        data.out_edges_map.resize(frames + 1);

        assert(min_seg_len >= 1);

        for (int i = 0; i < frames + 1; ++i) {
            for (int j = min_seg_len; j <= max_seg_len; ++j) {
                int tail = i;
                int head = i + j;

                if (head > frames) {
                    continue;
                }

                data.edges.push_back(lattice::edge_data {tail, head, 0, "<label>"});
                int e = data.edges.size() - 1;

                data.in_edges.at(head).push_back(e);
                data.in_edges_map.at(head)["<label>"].push_back(e);
                data.out_edges.at(tail).push_back(e);
                data.in_edges_map.at(tail)["<label>"].push_back(e);
            }
        }

        data.initials.push_back(0);
        data.finals.push_back(frames);

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(data));

        return f;
    }

    scrf_t make_graph_scrf(int frames, std::shared_ptr<lm::fst> lm, int min_seg_len, int max_seg_len)
    {
        scrf_t result;

        lattice::fst segmentation = make_segmentation_lattice(frames, min_seg_len, max_seg_len);
        lattice::add_eps_loops(segmentation);

        fst::composed_fst<lattice::fst, lm::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(segmentation);
        comp.fst2 = lm;

        result.fst = std::make_shared<decltype(comp)>(comp);

        result.topo_order = scrf::topo_order(result);

        return result;
    }

    scrf::scrf_t make_lat_scrf(lattice::fst lat, std::shared_ptr<lm::fst> lm)
    {
        scrf::scrf_t graph;

        lattice::fst_data data = *(lat.data);
        lattice::fst new_lat;
        new_lat.data = std::make_shared<lattice::fst_data>(data);

        lattice::add_eps_loops(new_lat);

        fst::composed_fst<lattice::fst, lm::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(new_lat);
        comp.fst2 = lm;
        graph.fst = std::make_shared<decltype(comp)>(comp);

        auto lm_v = lm->vertices();
        std::reverse(lm_v.begin(), lm_v.end());

        std::vector<std::tuple<int, int>> topo_order;
        for (auto v: lattice::topo_order(*(comp.fst1))) {
            for (auto u: lm_v) {
                topo_order.push_back(std::make_tuple(v, u));
            }
        }
        graph.topo_order = std::move(topo_order);

        return graph;
    }

    lm::fst make_ground_truth_lm(std::vector<std::string> const& labels)
    {
        lm::fst_data data;

        data.vertices = labels.size() + 1;
        data.in_edges.resize(data.vertices);
        data.out_edges.resize(data.vertices);
        data.in_edges_map.resize(data.vertices);
        data.out_edges_map.resize(data.vertices);

        int v = 0;
        for (int i = 0; i < labels.size(); ++i) {
            add_edge(data, i, v, v + 1, 0.0, "<label>", labels[i]);
            ++v;
        }

        data.initials.push_back(0);
        data.finals.push_back(v);

        return lm::fst { std::make_shared<lm::fst_data>(data) };
    }

    scrf_t make_forced_alignment_scrf(int frames,
        std::vector<std::string> const& labels, int min_seg_len, int max_seg_len)
    {
        scrf_t result;

        lattice::fst seg = make_segmentation_lattice(frames, min_seg_len, max_seg_len);

        lm::fst lm = make_ground_truth_lm(labels);

        fst::composed_fst<lattice::fst, lm::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(seg);
        comp.fst2 = std::make_shared<lm::fst>(lm);
        result.fst = std::make_shared<decltype(comp)>(comp);

        auto lm_v = lm.vertices();

        std::vector<std::tuple<int, int>> topo_order;
        for (auto v: lattice::topo_order(*(comp.fst1))) {
            for (auto u: lm_v) {
                topo_order.push_back(std::make_tuple(v, u));
            }
        }
        result.topo_order = std::move(topo_order);

        return result;
    }

    std::unordered_map<std::string, int> load_phone_id(std::string filename)
    {
        std::unordered_map<std::string, int> result;
        std::string line;
        std::ifstream ifs { filename };
    
        int i = 0;
        while (std::getline(ifs, line)) {
            result[line] = i;
            ++i;
        }
    
        return result;
    }

    namespace first_order {

        scrf_t make_graph_scrf(int frames,
            std::vector<int> const& labels,
            int min_seg_len, int max_seg_len)
        {
            ilat::fst_data data;

            for (int i = 0; i < frames + 1; ++i) {
                ilat::add_vertex(data, i, i);
            }

            assert(min_seg_len >= 1);

            for (int i = 0; i < frames + 1; ++i) {
                for (int j = min_seg_len; j <= max_seg_len; ++j) {
                    int tail = i;
                    int head = i + j;

                    if (head > frames) {
                        continue;
                    }

                    for (auto& ell: labels) {
                        ilat::add_edge(data, data.edges.size(), ell, tail, head, 0);
                    }
                }
            }

            data.initials.push_back(0);
            data.finals.push_back(frames);

            ilat::fst fst;
            fst.data = std::make_shared<ilat::fst_data>(std::move(data));

            scrf_t result;

            result.fst = std::make_shared<ilat::fst>(fst);
            result.topo_order = fst::topo_order(fst);

            return result;
        }

        fst::path<scrf_t> make_min_cost_path(
            scrf_t& min_cost,
            fst::path<scrf_t> const& gold_path)
        {
            min_cost.weight_func = std::make_shared<neg_cost>(
                neg_cost { std::make_shared<seg_cost>(make_overlap_cost(gold_path)) });
            min_cost.topo_order = fst::topo_order(*min_cost.fst);
            fst::path<scrf_t> min_cost_path = shortest_path(
                min_cost, min_cost.topo_order);
        
            double min_cost_path_weight = 0;
        
            std::cout << "min cost path: ";
            for (auto& e: min_cost_path.edges()) {
                int tail = min_cost_path.tail(e);
                int head = min_cost_path.head(e);
        
                std::cout << min_cost_path.output(e) << " ";
                min_cost_path_weight += min_cost_path.weight(e);
            }
            std::cout << std::endl;
        
            std::cout << "cost: " << min_cost_path_weight << std::endl;
        
            return min_cost_path;
        }
    
        fst::path<scrf_t> make_ground_truth_path(
            scrf_t& ground_truth)
        {
            return fst::make_path(ground_truth, ground_truth.edges());
        }

    }

}
