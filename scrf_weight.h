#ifndef SCRF_WEIGHT_H
#define SCRF_WEIGHT_H

#include "scrf/scrf.h"
#include "scrf/scrf_feat.h"

namespace scrf {

    struct composite_weight
        : public scrf_weight {

        std::vector<std::shared_ptr<scrf_weight>> weights;

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    std::shared_ptr<scrf_weight> operator+(std::shared_ptr<scrf_weight> w1,
        std::shared_ptr<scrf_weight> w2);

    namespace score {

        struct linear_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            linear_score(param_t const& param, std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct cached_linear_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            mutable std::unordered_map<std::tuple<int, int>, double> cache;

            cached_linear_score(param_t const& param, std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

    composite_weight make_weight(
        std::vector<std::string> const& features,
        param_t const& param,
        composite_feature const& feat);

    namespace first_order {

        struct composite_weight
            : public scrf_weight {

            std::vector<std::shared_ptr<scrf_weight>> weights;

            virtual real operator()(ilat::fst const& fst,
                int e) const override;

        };

        std::shared_ptr<scrf_weight> operator+(std::shared_ptr<scrf_weight> w1,
            std::shared_ptr<scrf_weight> w2);

        namespace score {

            struct linear_score
                : public scrf_weight {

                param_t const& param;
                std::shared_ptr<scrf_feature> feat;

                linear_score(param_t const& param, std::shared_ptr<scrf_feature> feat);

                virtual real operator()(ilat::fst const& fst,
                    int e) const override;

            };

            struct cached_linear_score
                : public scrf_weight {

                param_t const& param;
                std::shared_ptr<scrf_feature> feat;

                mutable std::vector<double> cache;
                mutable std::vector<bool> in_cache;

                cached_linear_score(param_t const& param, std::shared_ptr<scrf_feature> feat,
                    ilat::fst const& fst);

                virtual real operator()(ilat::fst const& fst,
                    int e) const override;

            };

        }

    }

    namespace experimental {

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

}


#endif
