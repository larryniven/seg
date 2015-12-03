#ifndef FEAT_H
#define FEAT_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/segfeat.h"

namespace scrf {

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        composite_feature(std::string name);

        virtual int size() const override;

        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    private:
        std::string name_;
        
    };

    struct lexicalized_feature
        : public scrf_feature {

        lexicalized_feature(
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<real>> const& frames);

        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<real>> const& frames;

        virtual int size() const override;
        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    namespace feature {

        struct lm_score
            : public scrf_feature {

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

}

#endif
