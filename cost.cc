#include "scrf/cost.h"

namespace scrf {

    real backoff_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        return fst.input(e) == "<eps>" ? -1 : 0;
    }

    overlap_cost::overlap_cost(fst::path<scrf_t> const& gold)
        : gold(gold)
    {}

    real overlap_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        int tail = std::get<0>(fst.tail(e));
        int head = std::get<0>(fst.head(e));

        int tail_time = fst.fst1->data->vertices.at(tail).time;
        int head_time = fst.fst1->data->vertices.at(head).time;

        if (tail == head) {
            return 0;
        }

        if (ebt::in(std::get<0>(e), edge_cache)) {
            
            int min_cost = std::numeric_limits<int>::max();

            for (auto& e_g: edge_cache.at(std::get<0>(e))) {
                int gold_tail = std::get<0>(gold.tail(e_g));
                int gold_head = std::get<0>(gold.head(e_g));

                int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
                int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

                int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);
                int union_ = std::max(gold_head_time, head_time) - std::min(gold_tail_time, tail_time);
                int cost = (gold.output(e_g) == fst.output(e) ? union_ - overlap : union_);

                if (gold.output(e_g) == "<s>" && fst.output(e) == "<s>"
                        || gold.output(e_g) == "</s>" && fst.output(e) == "</s>"
                        || gold.output(e_g) == "sil" && fst.output(e) == "sil") {

                    cost = head_time - tail_time - overlap;
                }

                if (cost < min_cost) {
                    min_cost = cost;
                }
            }

            if (edge_cache.at(std::get<0>(e)).size() == 0) {
                return head_time - tail_time;
            }

            return min_cost;

        }

        int max_overlap = 0;
        std::vector<std::tuple<int, int>> max_overlap_edges;

        for (auto& e_g: gold.edges()) {
            int gold_tail = std::get<0>(gold.tail(e_g));
            int gold_head = std::get<0>(gold.head(e_g));

            int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
            int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

            int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);

            if (overlap > max_overlap) {
                max_overlap = overlap;
                max_overlap_edges.clear();
                max_overlap_edges.push_back(e_g);
            } else if (overlap == max_overlap) {
                max_overlap_edges.push_back(e_g);
            }
        }

        int min_cost = std::numeric_limits<int>::max();

        for (auto& e_g: max_overlap_edges) {
            int gold_tail = std::get<0>(gold.tail(e_g));
            int gold_head = std::get<0>(gold.head(e_g));

            int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
            int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

            int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);
            int union_ = std::max(gold_head_time, head_time) - std::min(gold_tail_time, tail_time);
            int cost = (gold.output(e_g) == fst.output(e) ? union_ - overlap : union_);

            if (gold.output(e_g) == "<s>" && fst.output(e) == "<s>"
                    || gold.output(e_g) == "</s>" && fst.output(e) == "</s>"
                    || gold.output(e_g) == "sil" && fst.output(e) == "sil") {

                cost = head_time - tail_time - overlap;
            }

            if (cost < min_cost) {
                min_cost = cost;
            }
        }

        edge_cache[std::get<0>(e)] = std::move(max_overlap_edges);

        if (max_overlap_edges.size() == 0) {
            return head_time - tail_time;
        }

        return min_cost;
    }

    neg_cost::neg_cost(std::shared_ptr<scrf_weight> cost)
        : cost(cost)
    {}

    real neg_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        return -(*cost)(fst, e);
    }

}
