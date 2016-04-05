#ifndef LOSS_H
#define LOSS_H

#include "scrf/loss.h"
#include "scrf/scrf.h"

namespace scrf {

    struct hinge_loss
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        fst::path<scrf_t> graph_path;

        hinge_loss(fst::path<scrf_t> const& gold, scrf_t const& graph);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };

    struct log_loss
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;

        std::unordered_map<scrf_t::vertex, double> forward;
        std::unordered_map<scrf_t::vertex, double> backward;
        double logZ;

        std::unordered_map<scrf_t::vertex, param_t> forward_feat;
        std::unordered_map<scrf_t::vertex, param_t> backward_feat;

        log_loss(fst::path<scrf_t> const& gold,
            scrf_t const& graph);

        virtual double loss() override;
        virtual param_t param_grad() override;
    };

#if 0
    struct filtering_loss
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        real alpha;

        fst::path<scrf_t> graph_path;
        fst::forward_one_best<scrf_t> forward;
        fst::backward_one_best<scrf_t> backward;
        std::unordered_map<scrf_t::vertex, param_t> f_param;
        std::unordered_map<scrf_t::vertex, param_t> b_param;
        real threshold;

        filtering_loss(
            fst::path<scrf_t> const& gold,
            scrf_t const& graph,
            real alpha);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };
#endif

    struct hinge_loss_beam
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        fst::path<scrf_t> graph_path;
        int beam_width;

        hinge_loss_beam(fst::path<scrf_t> const& gold, scrf_t const& graph, int beam_width);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };

    namespace first_order {

        struct hinge_loss
            : public loss_func {

            fst::path<scrf_t> const& gold;
            scrf_t const& graph;
            fst::path<scrf_t> graph_path;

            hinge_loss(fst::path<scrf_t> const& gold, scrf_t const& graph);

            virtual real loss() override;
            virtual param_t param_grad() override;
        };

        struct log_loss
            : public loss_func {

            fst::path<scrf_t> const& gold;
            scrf_t const& graph;

            std::vector<double> forward;
            std::vector<double> backward;
            double logZ;

            std::vector<param_t> forward_feat;
            std::vector<param_t> backward_feat;

            log_loss(fst::path<scrf_t> const& gold,
                scrf_t const& graph);

            virtual double loss() override;
            virtual param_t param_grad() override;
        };

        struct filtering_loss
            : public loss_func {

            fst::path<scrf_t> const& gold;
            scrf_t const& graph;
            double alpha;

            fst::path<scrf_t> graph_path;
            fst::forward_one_best<scrf_t> forward;
            fst::backward_one_best<scrf_t> backward;
            std::vector<param_t> f_param;
            std::vector<param_t> b_param;
            double threshold;

            filtering_loss(
                fst::path<scrf_t> const& gold,
                scrf_t const& graph,
                double alpha);

            virtual double loss() override;
            virtual param_t param_grad() override;
        };
    }

}

#endif
