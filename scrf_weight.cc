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
            feat_t f;
            (*feat)(f, fst, e);
            real s = dot(param, to_param(std::move(f)));

            return s;
        }

        cached_linear_score::cached_linear_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real cached_linear_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(e, cache)) {
                return cache.at(e);
            }

            feat_t f;
            (*feat)(f, fst, e);
            real s = dot(param, to_param(std::move(f)));

            cache[e] = s;

            return s;
        }

    }

    composite_weight make_weight(
        std::vector<std::string> const& features,
        param_t const& param,
        composite_feature const& feat)
    {
        composite_weight result;

        score::linear_score rest_score { param,
            std::make_shared<composite_feature>(feat) };

        result.weights.push_back(std::make_shared<score::linear_score>(rest_score));

        return result;
    }

    namespace first_order {

        real composite_weight::operator()(ilat::fst const& fst,
            int e) const
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

            real linear_score::operator()(ilat::fst const& fst,
                int e) const
            {
                param_t f;
                (*feat)(f, fst, e);
                real s = dot(param, f);

                return s;
            }

            cached_linear_score::cached_linear_score(param_t const& param,
                    std::shared_ptr<scrf_feature> feat)
                : param(param), feat(feat)
            {}

            real cached_linear_score::operator()(ilat::fst const& fst,
                int e) const
            {
                if (e < in_cache.size() && in_cache[e]) {
                    return cache[e];
                }

                param_t f;
                (*feat)(f, fst, e);
                real s = dot(param, f);

                if (cache.size() == 0) {
                    in_cache.resize(fst.edges().size());
                    cache.resize(fst.edges().size());
                }
                cache[e] = s;
                in_cache[e] = 1;

                return s;
            }

        }

    }

}
