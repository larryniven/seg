#ifndef LOSS_H
#define LOSS_H

#include "seg/scrf.h"

namespace scrf {

    template <class scrf_data>
    struct hinge_loss
        : public loss_func_with_frame_grad<typename scrf_data_trait<scrf_data>::vector,
            typename scrf_data_trait<scrf_data>::base_fst> {

        using vector = typename scrf_data_trait<scrf_data>::vector;
        using fst = typename scrf_data_trait<scrf_data>::base_fst;

        scrf_data gold_path;
        scrf_data graph_path;

        hinge_loss(scrf_data const& gold_path, scrf_data const& graph);

        virtual double loss() const override;
        virtual vector param_grad() const override;

        virtual void frame_grad(
            scrf_feature_with_frame_grad<fst, vector> const& feat_func,
            std::vector<std::vector<double>>& grad,
            vector const& param) const override;
    };

}

namespace scrf {

    template <class scrf_data>
    hinge_loss<scrf_data>::hinge_loss(scrf_data const& gold_path, scrf_data const& graph)
        : gold_path(gold_path), graph_path(graph)
    {
        graph_path.fst = shortest_path(graph);

        if (graph_path.fst->edges().size() == 0) {
            std::cout << "no cost aug path" << std::endl;
            exit(1);
        }
    }

    template <class scrf_data>
    double hinge_loss<scrf_data>::loss() const
    {
        double gold_score = 0;

        for (auto& e: gold_path.fst->edges()) {
            gold_score += weight(gold_path, e);
        }

        double graph_score = 0;

        for (auto& e: graph_path.fst->edges()) {
            graph_score += weight(graph_path, e);
        }

        return graph_score - gold_score;
    }

    template <class scrf_data>
    typename scrf_data_trait<scrf_data>::vector hinge_loss<scrf_data>::param_grad() const
    {
        using vector = typename scrf_data_trait<scrf_data>::vector;

        vector result;

        for (auto& e: gold_path.fst->edges()) {
            vector f;
            feature(gold_path, f, e);

            isub(result, f);
        }

        for (auto& e: graph_path.fst->edges()) {
            vector f;
            feature(graph_path, f, e);

            iadd(result, f);
        }

        return result;
    }

    template <class scrf_data>
    void hinge_loss<scrf_data>::frame_grad(
        scrf_feature_with_frame_grad<typename scrf_data_trait<scrf_data>::base_fst,
            typename scrf_data_trait<scrf_data>::vector> const& feat_func,
        std::vector<std::vector<double>>& grad,
        typename scrf_data_trait<scrf_data>::vector const& param) const
    {
        using vector = typename scrf_data_trait<scrf_data>::vector;

        vector neg_param = param;
        imul(neg_param, -1);

        for (auto& e: gold_path.fst->edges()) {
            feat_func.frame_grad(grad, neg_param, *gold_path.fst, e);
        }

        for (auto& e: graph_path.fst->edges()) {
            feat_func.frame_grad(grad, param, *graph_path.fst, e);
        }
    }

}

#endif
