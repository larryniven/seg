#include "scrf/scrf_weight.h"
#include "scrf/scrf_feat.h"

namespace scrf {

    real composite_weight::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        real sum = 0;

        for (auto& w: weights) {
            sum += (*w)(fst, e);
        }

        return sum;
    }

    std::shared_ptr<scrf_weight> operator+(std::shared_ptr<scrf_weight> w1,
        std::shared_ptr<scrf_weight> w2)
    {
        composite_weight result;

        result.weights.push_back(w1);
        result.weights.push_back(w2);

        return std::make_shared<composite_weight>(result);
    }

    namespace score {

        linear_score::linear_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real linear_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            return s;
        }

        label_score::label_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real label_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(fst.output(e), cache)) {
                return cache[fst.output(e)];
            }

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            cache[fst.output(e)] = s;

            return s;
        }

        lm_score::lm_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real lm_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(std::get<1>(e), cache)) {
                return cache[std::get<1>(e)];
            }

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            cache[std::get<1>(e)] = s;

            return s;
        }

        lattice_score::lattice_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real lattice_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            /*
            if (ebt::in(std::make_tuple(std::get<0>(e), fst.output(e)), cache)) {
                return cache[std::make_tuple(std::get<0>(e), fst.output(e))];
            }
            */

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            // cache[std::make_tuple(std::get<0>(e), fst.output(e))] = s;

            return s;
        }

    }

    composite_weight make_weight(
        param_t const& param,
        composite_feature const& feat)
    {
        composite_weight result;

        composite_feature lattice_feat { "lattice-feat" };
        lattice_feat.features.push_back(feat.features[0]);
        lattice_feat.features.push_back(feat.features[1]);

        score::lattice_score lattice_score { param, std::make_shared<composite_feature>(lattice_feat) };
        score::lm_score lm_score { param, feat.features[2] };
        score::label_score label_score { param, feat.features[3] };
        score::linear_score rest_score { param, feat.features[4] };

        result.weights.push_back(std::make_shared<score::lattice_score>(lattice_score));
        result.weights.push_back(std::make_shared<score::lm_score>(lm_score));
        result.weights.push_back(std::make_shared<score::label_score>(label_score));
        result.weights.push_back(std::make_shared<score::linear_score>(rest_score));

        return result;
    }

}
