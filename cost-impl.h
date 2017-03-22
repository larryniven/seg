namespace cost {

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

}
