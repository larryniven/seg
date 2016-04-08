#ifndef FEAT_H
#define FEAT_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/segfeat.h"

namespace scrf {

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        composite_feature();

        virtual void operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    std::vector<double>& lexicalize(int order, feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e);

    struct segment_feature
        : public scrf_feature {

        segment_feature(
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<real>> const& frames);

        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<real>> const& frames;

        virtual void operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    namespace feature {

        struct lm_score
            : public scrf_feature {

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct external_feature
            : public scrf_feature {

            int order;
            std::vector<int> dims;

            external_feature(int order, std::vector<int> dims);

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct frame_feature
            : public scrf::scrf_feature {
        
            std::vector<std::vector<double>> const& frames;
            std::unordered_map<std::string, std::vector<int>> label_dim;
        
            frame_feature(std::vector<std::vector<double>> const& frames,
                std::unordered_map<std::string, std::string> const& args);

            std::unordered_map<std::string, std::vector<int>>
            load_label_dim(std::string filename);
        
            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        
        };

        struct quad_length
            : public scrf_feature {

            int order;
            std::unordered_map<std::string, double> mean;
            std::unordered_map<std::string, double> var;

            quad_length(int order, std::unordered_map<std::string, std::string> const& args);

            std::tuple<std::unordered_map<std::string, double>,
                std::unordered_map<std::string, double>>
            load_length_stat(std::string filename) const;

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

    namespace first_order {

        struct composite_feature
            : public scrf_feature {

            std::vector<std::shared_ptr<scrf_feature>> features;

            composite_feature();

            virtual void operator()(
                param_t& feat, ilat::fst const& fst, int e) const override;

        };

        la::vector<double>& lexicalize(feat_dim_alloc const& alloc,
            int order, param_t& feat, ilat::fst const& fst, int e);

        struct segment_feature
            : public scrf_feature {

            feat_dim_alloc& alloc;
            int dim;
            int order;
            std::shared_ptr<segfeat::la::feature> feat_func;
            std::vector<std::vector<real>> const& frames;

            segment_feature(
                feat_dim_alloc& alloc,
                int order,
                std::shared_ptr<segfeat::la::feature> raw_feat_func,
                std::vector<std::vector<real>> const& frames);

            virtual void operator()(
                param_t& feat, ilat::fst const& fst, int e) const override;
        };

        namespace feature {

            struct lattice_score
                : public scrf_feature {

                feat_dim_alloc& alloc;
                int dim;

                lattice_score(feat_dim_alloc& alloc);

                virtual void operator()(
                    param_t& feat, ilat::fst const& fst, int e) const override;

            };

            struct external_feature
                : public scrf_feature {

                feat_dim_alloc& alloc;
                int dim;
                int order;
                std::vector<int> dims;

                external_feature(feat_dim_alloc& alloc,
                    int order, std::vector<int> dims);

                virtual void operator()(
                    param_t& feat, ilat::fst const& fst, int e) const override;

            };

            struct frame_feature
                : public scrf_feature {
            
                feat_dim_alloc& alloc;
                int dim;
                std::vector<std::vector<double>> const& frames;
                std::unordered_map<std::string, int> label_id;
                std::vector<std::vector<int>> id_dim;
            
                frame_feature(feat_dim_alloc& alloc,
                    std::vector<std::vector<double>> const& frames,
                    std::unordered_map<std::string, std::string> const& args);
            
                std::vector<std::vector<int>>
                load_label_dim(std::string filename,
                    std::unordered_map<std::string, int> const& label_id);

                virtual void operator()(
                    param_t& feat, ilat::fst const& fst, int e) const override;
            
            };

            struct quad_length
                : public scrf_feature {

                feat_dim_alloc& alloc;
                int dim;
                int order;
                std::vector<double> mean;
                std::vector<double> var;
                std::unordered_map<std::string, int> label_id;
                std::vector<int> sils;

                quad_length(feat_dim_alloc& alloc, int order,
                    std::unordered_map<std::string, std::string> const& args);

                std::tuple<std::vector<double>, std::vector<double>>
                load_length_stat(std::string filename,
                    std::unordered_map<std::string, int> const& label_id) const;

                virtual void operator()(
                    param_t& feat, ilat::fst const& fst, int e) const override;

            };

            struct max_hits
                : public scrf_feature {

                feat_dim_alloc& alloc;
                int dim;
                int order;
                double percentile;
                int nhits;
                std::vector<std::vector<double>> const& frames;
                std::vector<double> max;
                std::vector<double> min;

                max_hits(feat_dim_alloc& alloc, int order, double percentile,
                    int nhits, std::vector<std::vector<double>> const& frames);

                virtual void operator()(
                    param_t& feat, ilat::fst const& fst, int e) const override;

            };

        }

    }

    namespace experimental {

        struct feat_dim_alloc {

            std::vector<int> order_dim;
            std::vector<int> const& labels;

            feat_dim_alloc(std::vector<int> const& labels);

            int alloc(int order, int dim);
        };

        template <class fst, class vector>
        struct lexicalizer {
            virtual double* operator()(feat_dim_alloc const& alloc,
                int order, vector& f, fst const& g, typename fst::edge e) const = 0;
        };

        template <class fst, class vector, class lexicalizer>
        struct segment_feature
            : public scrf_feature<fst, vector> {

            segment_feature(
                feat_dim_alloc& alloc,
                int order,
                std::shared_ptr<segfeat::experimental::feature> raw_feat_func,
                std::vector<std::vector<double>> const& frames);

            feat_dim_alloc& alloc;
            int dim;
            int order;
            std::shared_ptr<segfeat::experimental::feature> feat_func;
            std::vector<std::vector<double>> const& frames;

            virtual void operator()(vector& f, fst const& g,
                typename fst::edge e) const override;
        };

        template <class fst, class vector>
        struct composite_feature
            : public scrf_feature<fst, vector> {

            std::vector<std::shared_ptr<scrf_feature<fst, vector>>> features;

            composite_feature();

            virtual void operator()(vector& feat, fst const& f,
                typename fst::edge e) const override;
        };

        template <class fst, class vector, class lexicalizer>
        segment_feature<fst, vector, lexicalizer>::segment_feature(
            feat_dim_alloc& alloc,
            int order,
            std::shared_ptr<segfeat::experimental::feature> feat_func,
            std::vector<std::vector<double>> const& frames)
            : alloc(alloc), order(order), feat_func(feat_func), frames(frames)
        {
            dim = alloc.alloc(order, feat_func->dim(frames.front().size()));
        }

        template <class fst, class vector, class lexicalizer>
        void segment_feature<fst, vector, lexicalizer>::operator()(
            vector& f, fst const& a, typename fst::edge e) const
        {
            double *g = lexicalizer()(alloc, order, f, a, e);

            (*feat_func)(g + dim, frames, a.time(a.tail(e)), a.time(a.head(e)));
        }

        template <class fst, class vector>
        composite_feature<fst, vector>::composite_feature()
        {}

        template <class fst, class vector>
        void composite_feature<fst, vector>::operator()(
            vector& feat, fst const& f,
            typename fst::edge e) const
        {
            for (auto& h: features) {
                (*h)(feat, f, e);
            }
        }
        
    }

}

#endif
