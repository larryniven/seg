#ifndef FEAT_H
#define FEAT_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/segfeat.h"

namespace scrf {

    struct feat_dim_alloc {

        std::vector<int> order_dim;
        std::vector<int> const& labels;

        feat_dim_alloc(std::vector<int> const& labels);

        int alloc(int order, int dim);
    };

    template <class fst, class vector>
    struct lexicalizer {
        virtual double* lex(feat_dim_alloc const& alloc,
            int order, vector& f, fst const& a, typename fst::edge e) const = 0;

        virtual double const* const_lex(feat_dim_alloc const& alloc,
            int order, vector const& f, fst const& a, typename fst::edge e)
        {
            return lex(alloc, order, const_cast<vector&>(f), a, e);
        }
    };

    template <class fst, class vector, class lexicalizer>
    struct segment_feature
        : public scrf_feature<fst, vector> {

        segment_feature(
            feat_dim_alloc& alloc,
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<double>> const& frames);

        feat_dim_alloc& alloc;
        int dim;
        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<double>> const& frames;

        virtual void operator()(vector& f, fst const& g,
            typename fst::edge e) const override;

    };

    template <class fst, class vector, class lexicalizer>
    struct segment_feature_with_frame_grad
        : public scrf_feature_with_frame_grad<fst, vector> {

        segment_feature_with_frame_grad(
            feat_dim_alloc& alloc,
            int order,
            std::shared_ptr<segfeat::feature_with_frame_grad> raw_feat_func,
            std::vector<std::vector<double>> const& frames);

        feat_dim_alloc& alloc;
        int dim;
        int order;
        std::shared_ptr<segfeat::feature_with_frame_grad> feat_func;
        std::vector<std::vector<double>> const& frames;

        virtual void operator()(vector& f, fst const& g,
            typename fst::edge e) const override;

        virtual void frame_grad(
            std::vector<std::vector<double>>& grad,
            vector const& feat_grad,
            fst const& a, typename fst::edge e) const;
    };

    template <class fst, class vector>
    struct composite_feature
        : public scrf_feature<fst, vector> {

        std::vector<std::shared_ptr<scrf_feature<fst, vector>>> features;

        virtual void operator()(vector& feat, fst const& f,
            typename fst::edge e) const override;
    };

    template <class fst, class vector>
    struct composite_feature_with_frame_grad
        : public scrf_feature_with_frame_grad<fst, vector> {

        std::vector<std::shared_ptr<scrf_feature_with_frame_grad<fst, vector>>> features;

        virtual void operator()(vector& feat, fst const& f,
            typename fst::edge e) const override;

        virtual void frame_grad(
            std::vector<std::vector<double>>& grad,
            vector const& param,
            fst const& a, typename fst::edge e) const;
    };

    template <class fst, class vector, class lexicalizer>
    segment_feature<fst, vector, lexicalizer>::segment_feature(
        feat_dim_alloc& alloc,
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<double>> const& frames)
        : alloc(alloc), order(order), feat_func(feat_func), frames(frames)
    {
        dim = alloc.alloc(order, feat_func->dim(frames.front().size()));
    }

    template <class fst, class vector, class lexicalizer>
    void segment_feature<fst, vector, lexicalizer>::operator()(
        vector& f, fst const& a, typename fst::edge e) const
    {
        double *g = lexicalizer().lex(alloc, order, f, a, e);

        (*feat_func)(g + dim, frames, a.time(a.tail(e)), a.time(a.head(e)));
    }

    template <class fst, class vector, class lexicalizer>
    segment_feature_with_frame_grad<fst, vector, lexicalizer>::segment_feature_with_frame_grad(
        feat_dim_alloc& alloc,
        int order,
        std::shared_ptr<segfeat::feature_with_frame_grad> feat_func,
        std::vector<std::vector<double>> const& frames)
        : alloc(alloc), order(order), feat_func(feat_func), frames(frames)
    {
        dim = alloc.alloc(order, feat_func->dim(frames.front().size()));
    }

    template <class fst, class vector, class lexicalizer>
    void segment_feature_with_frame_grad<fst, vector, lexicalizer>::operator()(
        vector& f, fst const& a, typename fst::edge e) const
    {
        double *g = lexicalizer().lex(alloc, order, f, a, e);

        (*feat_func)(g + dim, frames, a.time(a.tail(e)), a.time(a.head(e)));
    }

    template <class fst, class vector, class lexicalizer>
    void segment_feature_with_frame_grad<fst, vector, lexicalizer>::frame_grad(
        std::vector<std::vector<double>>& grad,
        vector const& feat_grad,
        fst const& a, typename fst::edge e) const
    {
        double const *g = lexicalizer().const_lex(alloc, order, feat_grad, a, e);

        feat_func->frame_grad(grad, g + dim, frames, a.time(a.tail(e)), a.time(a.head(e)));
    }

    template <class fst, class vector>
    void composite_feature<fst, vector>::operator()(
        vector& feat, fst const& f,
        typename fst::edge e) const
    {
        for (auto& h: features) {
            (*h)(feat, f, e);
        }
    }

    template <class fst, class vector>
    void composite_feature_with_frame_grad<fst, vector>::operator()(
        vector& feat, fst const& f,
        typename fst::edge e) const
    {
        for (auto& h: features) {
            (*h)(feat, f, e);
        }
    }

    template <class fst, class vector>
    void composite_feature_with_frame_grad<fst, vector>::frame_grad(
        std::vector<std::vector<double>>& grad,
        vector const& param,
        fst const& a, typename fst::edge e) const
    {
        for (auto& h: features) {
            h->frame_grad(grad, param, a, e);
        }
    }
}

#endif
