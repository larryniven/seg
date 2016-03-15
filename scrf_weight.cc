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

            feat_t f;
            (*feat)(f, fst, e);
            real s = dot(param, to_param(std::move(f)));

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

            feat_t f;
            (*feat)(f, fst, e);
            real s = dot(param, to_param(std::move(f)));

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
            std::tuple<int, std::string> key = std::make_tuple(std::get<0>(e), fst.output(e));

            if (ebt::in(key, cache)) {
                return cache.at(key);
            }

            feat_t f;
            (*feat)(f, fst, e);
            real s = dot(param, to_param(std::move(f)));

            cache[key] = s;

            return s;
        }

    }

    composite_weight make_weight(
        std::vector<std::string> const& features,
        param_t const& param,
        composite_feature const& feat)
    {
        composite_weight result;

        composite_feature lattice_feat;
        composite_feature lm_feat;
        composite_feature rest_feat;

        for (int i = 0; i < features.size(); ++i) {
            std::vector<std::string> parts = ebt::split("@");
            int order = 0;
            if (parts.size() > 1) {
                order = std::stoi(parts[1]);
            }

            if (order <= 1) {
                lattice_feat.features.push_back(feat.features[i]);
            } else if (ebt::startswith(features[i], "lm-score")) {
                lm_feat.features.push_back(feat.features[i]);
            } else {
                rest_feat.features.push_back(feat.features[i]);
            }
        }

        score::lattice_score lattice_score { param,
            std::make_shared<composite_feature>(lattice_feat) };
        score::lm_score lm_score { param,
            std::make_shared<composite_feature>(lm_feat) };
        score::linear_score rest_score { param,
            std::make_shared<composite_feature>(rest_feat) };

        result.weights.push_back(std::make_shared<score::lattice_score>(lattice_score));
        result.weights.push_back(std::make_shared<score::lm_score>(lm_score));
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

        }

    }

}
