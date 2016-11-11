#ifndef SEG_COST_H
#define SEG_COST_H

#include <vector>
#include "speech/speech.h"

namespace segcost {

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
    struct hit_cost
        : public cost<symbol> {

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

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
    struct overlap_portion_cost
        : public cost<symbol> {

        std::vector<symbol> sils;

        overlap_portion_cost();
        overlap_portion_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

    template <class symbol>
    struct cover_cost
        : public cost<symbol> {

        std::vector<symbol> sils;

        cover_cost();
        cover_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

}

namespace segcost {

    template <class symbol>
    double hit_cost<symbol>::operator()(std::vector<segment<symbol>> const& gold_segs,
        segment<symbol> const& e) const
    {
        if (e.start_time == e.end_time && e.label == 0) {
            return 0.01;
        }

        if (e.start_time == e.end_time) {
            return 0;
        }

        segment<symbol> max_seg;
        int max_overlap = -1;

        int left = 0;
        int right = gold_segs.size() - 1;

        while (left + 1 < right) {
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

        while (left + 1 < right) {
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

            if (s.label == e.label) {
                return 0;
            }
        }

        return 1;
    }

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
        if (e.start_time == e.end_time && e.label == 0) {
            return 0.01;
        }

        if (e.start_time == e.end_time) {
            return 0;
        }

        segment<symbol> max_seg;
        int max_overlap = -1;

        int left = 0;
        int right = gold_segs.size() - 1;

        while (left + 1 < right) {
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

        while (left + 1 < right) {
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

    template <class symbol>
    overlap_portion_cost<symbol>::overlap_portion_cost()
    {}

    template <class symbol>
    overlap_portion_cost<symbol>::overlap_portion_cost(std::vector<symbol> sils)
        : sils(sils)
    {}

    template <class symbol>
    double overlap_portion_cost<symbol>::operator()(std::vector<segment<symbol>> const& gold_segs,
        segment<symbol> const& e) const
    {
        if (e.start_time == e.end_time && e.label == 0) {
            return 0.001;
        }

        if (e.start_time == e.end_time) {
            return 0;
        }

        segment<symbol> max_seg;
        int max_overlap = -1;

        int left = 0;
        int right = gold_segs.size() - 1;

        while (left + 1 < right) {
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

        while (left + 1 < right) {
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
                return 1 - max_overlap / double(e.end_time - e.start_time);
            }
        }

        if (e.label == max_seg.label) {
            return 1 - max_overlap / double(union_);
        } else {
            return 1;
        }
    }

    template <class symbol>
    cover_cost<symbol>::cover_cost()
    {}

    template <class symbol>
    cover_cost<symbol>::cover_cost(std::vector<symbol> sils)
        : sils(sils)
    {}

    template <class symbol>
    double cover_cost<symbol>::operator()(std::vector<segment<symbol>> const& gold_segs,
        segment<symbol> const& e) const
    {
        if (e.start_time == e.end_time && e.label == 0) {
            return 0.001;
        }

        if (e.start_time == e.end_time) {
            return 0;
        }

        segment<symbol> max_seg;
        int max_overlap = -1;

        int left = 0;
        int right = gold_segs.size() - 1;

        while (left + 1 < right) {
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

        while (left + 1 < right) {
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

        if (e.label == max_seg.label) {
            return 1 - max_overlap / double(e.end_time - e.start_time);
        } else {
            return 1 - 0.5 * max_overlap / double(e.end_time - e.start_time);
        }
    }

}

#endif
