#ifndef LOSS_H
#define LOSS_H

#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf.h"

namespace scrf {

    template <class fst, class vector, class path_maker>
    struct hinge_loss
        : public loss_func<vector> {

        fst const& gold;
        fst const& graph;
        std::shared_ptr<fst> graph_path;

        hinge_loss(fst const& gold, fst const& graph);

        virtual double loss() const override;
        virtual vector param_grad() const override;
    };

    template <class fst, class vector, class path_maker>
    struct hinge_loss_with_frame_grad
        : public loss_func_with_frame_grad<vector> {

        fst const& gold;
        fst const& graph;
        std::shared_ptr<fst> graph_path;

        hinge_loss_with_frame_grad(fst const& gold, fst const& graph);

        virtual double loss() const override;
        virtual vector param_grad() const override;

        virtual void frame_grad(std::vector<std::vector<double>>& grad,
            vector const& param) const override;
    };

    template <class fst, class vector, class path_maker>
    hinge_loss<fst, vector, path_maker>::hinge_loss(fst const& gold, fst const& graph)
        : gold(gold), graph(graph)
    {
        graph_path = ::fst::shortest_path<fst, path_maker>(graph);

        if (graph_path->edges().size() == 0) {
            std::cout << "no cost aug path" << std::endl;
            exit(1);
        }
    }

    template <class fst, class vector, class path_maker>
    double hinge_loss<fst, vector, path_maker>::loss() const
    {
        double gold_score = 0;

        for (auto& e: gold.edges()) {
            gold_score += gold.weight(e);
        }

        double graph_score = 0;

        for (auto& e: graph_path->edges()) {
            graph_score += graph_path->weight(e);
        }

        return graph_score - gold_score;
    }

    template <class fst, class vector, class path_maker>
    vector hinge_loss<fst, vector, path_maker>::param_grad() const
    {
        vector result;

        for (auto& e: gold.edges()) {
            vector f;
            gold.feature(f, e);

            isub(result, f);
        }

        for (auto& e: graph_path->edges()) {
            vector f;
            graph.feature(f, e);

            iadd(result, f);
        }

        return result;
    }

    template <class fst, class vector, class path_maker>
    hinge_loss_with_frame_grad<fst, vector, path_maker>::hinge_loss_with_frame_grad(
            fst const& gold, fst const& graph)
        : gold(gold), graph(graph)
    {
        graph_path = ::fst::shortest_path<fst, path_maker>(graph);

        if (graph_path->edges().size() == 0) {
            std::cout << "no cost aug path" << std::endl;
            exit(1);
        }
    }

    template <class fst, class vector, class path_maker>
    double hinge_loss_with_frame_grad<fst, vector, path_maker>::loss() const
    {
        double gold_score = 0;

        for (auto& e: gold.edges()) {
            gold_score += gold.weight(e);
        }

        double graph_score = 0;

        for (auto& e: graph_path->edges()) {
            graph_score += graph_path->weight(e);
        }

        return graph_score - gold_score;
    }

    template <class fst, class vector, class path_maker>
    vector hinge_loss_with_frame_grad<fst, vector, path_maker>::param_grad() const
    {
        vector result;

        for (auto& e: gold.edges()) {
            vector f;
            gold.feature(f, e);

            isub(result, f);
        }

        for (auto& e: graph_path->edges()) {
            vector f;
            graph.feature(f, e);

            iadd(result, f);
        }

        return result;
    }

    template <class fst, class vector, class path_maker>
    void hinge_loss_with_frame_grad<fst, vector, path_maker>::frame_grad(
        std::vector<std::vector<double>>& grad, vector const& param) const
    {
        for (auto& e: gold.edges()) {
            gold.frame_grad(grad, param, e);
        }

        for (int i = 0; i < grad.size(); ++i) {
            for (int j = 0; j < grad[i].size(); ++j) {
                 grad[i][j] = -grad[i][j];
            }
        }

        for (auto& e: graph_path->edges()) {
            graph.frame_grad(grad, param, e);
        }
    }

}

#endif
