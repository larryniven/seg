namespace seg {

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
    void composite_weight<fst>::accumulate_grad(double g, fst const& f,
        typename fst::edge e) const
    {
        for (auto& w: weights) {
            w->accumulate_grad(g, f, e);
        }
    }

    template <class fst>
    void composite_weight<fst>::grad() const
    {
        for (auto& w: weights) {
            w->grad();
        }
    }

}
