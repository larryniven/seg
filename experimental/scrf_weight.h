#ifndef SCRF_WEIGHT_H
#define SCRF_WEIGHT_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"

namespace scrf {

    template <class fst>
    struct composite_weight
        : public scrf_weight<fst> {

        std::vector<std::shared_ptr<scrf_weight<fst>>> weights;

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;

    };

    template <class fst>
    struct mul
        : public scrf_weight<fst> {

        std::shared_ptr<scrf_weight<fst>> weight;
        double alpha;

        mul(std::shared_ptr<scrf_weight<fst>> const& w, double a);

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;
    };

    template <class fst, class vector>
    struct linear_score
        : public scrf_weight<fst> {

        vector const& param;
        std::shared_ptr<scrf_feature<fst, vector>> feat;

        linear_score(vector const& param, std::shared_ptr<scrf_feature<fst, vector>> feat);

        virtual double operator()(fst const& f, typename fst::edge e) const override;

    };

    template <class fst, class vector>
    struct cached_linear_score
        : public scrf_weight<fst> {

        vector const& param;
        std::shared_ptr<scrf_feature<fst, vector>> feat;

        mutable std::unordered_map<typename fst::edge, double> cache;

        cached_linear_score(vector const& param, std::shared_ptr<scrf_feature<fst, vector>> feat);

        virtual double operator()(fst const& f, typename fst::edge e) const override;

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
    mul<fst>::mul(std::shared_ptr<scrf_weight<fst>> const& w, double a)
        : weight(w), alpha(a)
    {}

    template <class fst>
    double mul<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        return (*weight)(f, e) * alpha;
    }

    template <class fst, class vector>
    linear_score<fst, vector>::linear_score(vector const& param,
            std::shared_ptr<scrf_feature<fst, vector>> feat)
        : param(param), feat(feat)
    {}

    template <class fst, class vector>
    double linear_score<fst, vector>::operator()(fst const& a, typename fst::edge e) const
    {
        vector f;
        (*feat)(f, a, e);
        return dot(param, f);

        /*

        for (int i = 0; i < feature.size(); ++i) {
            (*feature[i])(v[feat->order()], a, e);
            la::weak_vector<double> d = lex.lex(alloc, i, param, a, e);
            la::dot(v[feat->order()], d)
        }

        */
    }

    template <class fst, class vector>
    cached_linear_score<fst, vector>::cached_linear_score(vector const& param,
            std::shared_ptr<scrf_feature<fst, vector>> feat)
        : param(param), feat(feat)
    {}

    template <class fst, class vector>
    double cached_linear_score<fst, vector>::operator()(fst const& a, typename fst::edge e) const
    {
        if (ebt::in(e, cache)) {
            return cache[e];
        }

        vector f;
        (*feat)(f, a, e);
        cache[e] = dot(param, f);
        return cache[e];
    }

}


#endif
