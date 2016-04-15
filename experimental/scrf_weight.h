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
    struct neg
        : public scrf_weight<fst> {

        std::shared_ptr<scrf_weight<fst>> weight;

        neg(std::shared_ptr<scrf_weight<fst>> weight);

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
    neg<fst>::neg(std::shared_ptr<scrf_weight<fst>> weight)
        : weight(weight)
    {}

    template <class fst>
    double neg<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        return -(*weight)(f, e);
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
    }

}


#endif
