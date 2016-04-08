#ifndef SEG_COST_H
#define SEG_COST_H

#include <vector>
#include "speech/speech.h"

namespace segcost {

    struct cost {

        virtual ~cost();

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const = 0;

    };

    struct overlap_cost
        : public cost {

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const override;

    };

    namespace first_order {

        struct segment {
            long start_time;
            long end_time;
            int label;
        };

        struct cost {

            virtual ~cost();

            virtual double operator()(std::vector<segment> const& gold_edges,
                segment const& e) const = 0;

        };

        struct overlap_cost
            : public cost {

            std::vector<int> sils;

            overlap_cost();
            overlap_cost(std::vector<int> sils);

            virtual double operator()(std::vector<segment> const& gold_edges,
                segment const& e) const override;

        };

    }

    namespace experimental {

        template <class symbol>
        struct segment {
            long start_time;
            long end_time;
            symbol label;
        };

        template <class symbol>
        struct cost {
            
            virtual ~cost()
            {}

            virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
                segment<symbol> const& e) const = 0;

        };

        template <class symbol>
        struct overlap_cost
            : public cost<symbol> {

            std::vector<symbol> sils;

            overlap_cost();
            overlap_cost(std::vector<symbol> sils);

            virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
                segment<symbol> const& e) const override;

        };

        template <class symbol>
        overlap_cost<symbol>::overlap_cost()
        {}

        template <class symbol>
        overlap_cost<symbol>::overlap_cost(std::vector<symbol> sils)
            : sils(sils)
        {}

        template <class symbol>
        double overlap_cost<symbol>::operator()(std::vector<segment<symbol>> const& gold_segs,
            segment<symbol> const& e) const
        {
            if (e.start_time == e.end_time) {
                return 0;
            }

            segment<symbol> max_seg;
            int max_overlap = -1;

            int left = 0;
            int right = gold_segs.size() - 1;

            while (left + 1 != right) {
                int mid = int((left + right) / 2);

                if (e.start_time == gold_segs[mid].start_time) {
                    left = mid;
                    break;
                } else if (e.start_time > gold_segs[mid].start_time) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            int start_left = left;

            left = 0;
            right = gold_segs.size() - 1;

            while (left + 1 != right) {
                int mid = int((left + right) / 2);

                if (e.end_time == gold_segs[mid].end_time) {
                    left = mid;
                    break;
                } else if (e.end_time > gold_segs[mid].end_time) {
                    left = mid;
                } else {
                    right = mid;
                }
            }

            int end_right = right;

            for (int i = start_left; i <= end_right; ++i) {
                auto& s = gold_segs[i];

                int overlap = std::max<int>(0, std::min(s.end_time, e.end_time)
                    - std::max(s.start_time, e.start_time));

                if (overlap > max_overlap) {
                    max_seg = s;
                    max_overlap = overlap;
                }
            }

            int union_ = std::max(max_seg.end_time, e.end_time)
                - std::min(max_seg.start_time, e.start_time);

            for (auto& s: sils) {
                if (e.label == s && e.label == max_seg.label) {
                    assert(e.end_time - e.start_time - max_overlap >= 0);
                    return (e.end_time - e.start_time) - max_overlap;
                }
            }

            if (e.label == max_seg.label) {
                return union_ - max_overlap;
            } else {
                return union_;
            }
        }

    }

}

#endif
