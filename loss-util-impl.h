namespace seg {

    template <class fst_type>
    void forward_exp_score<fst_type>::merge(fst_type const& f, std::vector<vertex> order,
        double logZ,
        std::unordered_map<vertex, double> const& forward_log_sum)
    {
        for (auto& i: f.initials()) {
            extra[i] = 0;
        }

        for (auto& i: order) {
            double sum = 0;

            for (auto& e: f.in_edges(i)) {
                sum += std::exp(f.weight(e) - logZ) * (extra.at(f.tail(e))
                    + f.weight(e) * std::exp(forward_log_sum.at(i)));
            }

            extra[i] = sum;
        }
    }

    template <class fst_type>
    void backward_exp_score<fst_type>::merge(fst_type const& f, std::vector<vertex> order,
        double logZ,
        std::unordered_map<vertex, double> const& backward_log_sum)
    {
        for (auto& i: f.finals()) {
            extra[i] = 0;
        }

        for (auto& i: order) {
            double sum = 0;

            for (auto& e: f.out_edges(i)) {
                sum += std::exp(f.weight(e) - logZ) * (extra.at(f.head(e))
                    + f.weight(e) * std::exp(backward_log_sum.at(i)));
            }

            extra[i] = sum;
        }
    }

}
