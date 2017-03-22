
namespace seg {

    template <class fst_t>
    cost_t<fst_t>::cost_t(std::shared_ptr<cost::cost_t<typename fst_t::output_symbol>> cost_func,
        std::vector<cost::segment<typename fst_t::output_symbol>> const& gold_segs)
        : cost_func(cost_func), gold_segs(gold_segs)
    {}

    template <class fst_t>
    double cost_t<fst_t>::operator()(fst_t const& f,
        typename fst_t::edge e) const
    {
        auto tail = f.tail(e);
        auto head = f.head(e);

        return (*cost_func)(gold_segs, cost::segment<typename fst_t::output_symbol> {
            f.time(tail), f.time(head), f.output(e) });
    }

    template <class fst_t>
    cost_t<fst_t> make_overlap_cost(
        std::vector<cost::segment<typename fst_t::output_symbol>> const& gold_segs,
        std::vector<typename fst_t::output_symbol> sils)
    {
        return cost_t<fst_t> { std::make_shared<cost::overlap_cost<typename fst_t::output_symbol>>(
            cost::overlap_cost<typename fst_t::output_symbol>{ sils }), gold_segs };
    }

}
