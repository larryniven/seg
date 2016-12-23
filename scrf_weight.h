#ifndef SCRF_WEIGHT_H
#define SCRF_WEIGHT_H

#include "seg/scrf.h"

namespace scrf {

    template <class fst>
    struct composite_weight
        : public scrf_weight<fst> {

        std::vector<std::shared_ptr<scrf_weight<fst>>> weights;

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;

        virtual void accumulate_grad(double g, fst const& f,
            typename fst::edge e) const override;

        virtual void grad() const override;

    };

    template <class fst>
    struct cached_weight
        : public scrf_weight<fst> {

        std::shared_ptr<scrf_weight<fst>> weight_func;

        mutable std::unordered_map<typename fst::edge, double> weights;

        cached_weight(std::shared_ptr<scrf_weight<fst>> weight_func);

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;

    };

    template <class fst>
    struct mul
        : public scrf_weight<fst> {

        std::shared_ptr<scrf_weight<fst>> weight;
        double alpha;

        mul(std::shared_ptr<scrf_weight<fst>> w, double a);

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;
    };

    template <class fst>
    double composite_weight<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        double sum = 0;

        for (auto& w: weights) {
            sum += (*w)(f, e);
        }

        return sum;
    }

    template <class fst>
    void composite_weight<fst>::accumulate_grad(double g, fst const& f,
        typename fst::edge e) const
    {
        for (auto& w: weights) {
            w->accumulate_grad(g, f, e);
        }
    }

    template <class fst>
    void composite_weight<fst>::grad() const
    {
        for (auto& w: weights) {
            w->grad();
        }
    }

    template <class fst>
    cached_weight<fst>::cached_weight(std::shared_ptr<scrf_weight<fst>> weight_func)
        : weight_func(weight_func)
    {}

    template <class fst>
    double cached_weight<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        if (!ebt::in(e, weights)) {
            weights[e] = (*weight_func)(f, e);
        }

        return weights[e];
    }

    template <class fst>
    mul<fst>::mul(std::shared_ptr<scrf_weight<fst>> w, double a)
        : weight(w), alpha(a)
    {}

    template <class fst>
    double mul<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        return (*weight)(f, e) * alpha;
    }

}


#endif
