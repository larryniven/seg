namespace seg {

    template <class fst_type>
    forward_exp_risk<fst_type>::forward_exp_risk(
        std::shared_ptr<risk_func<fst_type>> risk)
        : risk(risk)
    {}

    template <class fst_type>
    void forward_exp_risk<fst_type>::merge(fst_type const& f, std::vector<vertex> order,
        std::unordered_map<vertex, double> const& forward_log_sum)
    {
        for (auto& i: f.initials()) {
            extra[i] = 0;
        }

        for (auto& i: order) {
            double sum = 0;

            for (auto& e: f.in_edges(i)) {
                if (!ebt::in(f.tail(e), forward_log_sum)) {
                    continue;
                }

                sum += std::exp(f.weight(e) + forward_log_sum.at(f.tail(e)) - forward_log_sum.at(f.head(e)))
                    * ((*risk)(f, e) + extra.at(f.tail(e)));
            }

            extra[i] = sum;
        }
    }

    template <class fst_type>
    backward_exp_risk<fst_type>::backward_exp_risk(
        std::shared_ptr<risk_func<fst_type>> risk)
        : risk(risk)
    {}

    template <class fst_type>
    void backward_exp_risk<fst_type>::merge(fst_type const& f, std::vector<vertex> order,
        std::unordered_map<vertex, double> const& backward_log_sum)
    {
        for (auto& i: f.finals()) {
            extra[i] = 0;
        }

        for (auto& i: order) {
            double sum = 0;

            for (auto& e: f.out_edges(i)) {
                if (!ebt::in(f.head(e), backward_log_sum)) {
                    continue;
                }

                sum += std::exp(f.weight(e) + backward_log_sum.at(f.head(e)) - backward_log_sum.at(f.tail(e)))
                    * ((*risk)(f, e) + extra.at(f.head(e)));
            }

            extra[i] = sum;
        }
    }

}
