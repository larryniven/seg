#include "scrf/scrf_util.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_weight.h"

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
        min_cost.weight_func = std::make_shared<scrf::neg_cost>(
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
    
    fst::path<scrf::scrf_t> make_ground_truth_path(
        scrf::scrf_t& ground_truth)
    {
        ground_truth.weight_func = std::make_shared<scrf::backoff_cost>(scrf::backoff_cost{});
        ground_truth.topo_order = scrf::topo_order(ground_truth);
        return scrf::shortest_path(ground_truth, ground_truth.topo_order);
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

}
