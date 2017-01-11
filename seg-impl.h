namespace seg {

    template <class seg_data>
    std::vector<typename seg_data::vertex> const&
    seg_fst<seg_data>::vertices() const
    {
        return data.fst->vertices();
    }

    template <class seg_data>
    std::vector<typename seg_data::edge> const&
    seg_fst<seg_data>::edges() const
    {
        return data.fst->edges();
    }

    template <class seg_data>
    typename seg_data::vertex
    seg_fst<seg_data>::head(typename seg_data::edge e) const
    {
        return data.fst->head(e);
    }

    template <class seg_data>
    typename seg_data::vertex
    seg_fst<seg_data>::tail(typename seg_data::edge e) const
    {
        return data.fst->tail(e);
    }

    template <class seg_data>
    std::vector<typename seg_data::edge> const&
    seg_fst<seg_data>::in_edges(typename seg_data::vertex v) const
    {
        return data.fst->in_edges(v);
    }

    template <class seg_data>
    std::vector<typename seg_data::edge> const&
    seg_fst<seg_data>::out_edges(typename seg_data::vertex v) const
    {
        return data.fst->out_edges(v);
    }

    template <class seg_data>
    double seg_fst<seg_data>::weight(typename seg_data::edge e) const
    {
        return (*data.weight_func)(*data.fst, e);
    }

    template <class seg_data>
    typename seg_data::input_symbol const&
    seg_fst<seg_data>::input(typename seg_data::edge e) const
    {
        return data.fst->input(e);
    }

    template <class seg_data>
    typename seg_data::output_symbol const&
    seg_fst<seg_data>::output(typename seg_data::edge e) const
    {
        return data.fst->output(e);
    }

    template <class seg_data>
    std::vector<typename seg_data::vertex> const&
    seg_fst<seg_data>::initials() const
    {
        return data.fst->initials();
    }

    template <class seg_data>
    std::vector<typename seg_data::vertex> const&
    seg_fst<seg_data>::finals() const
    {
        return data.fst->finals();
    }

    template <class seg_data>
    long seg_fst<seg_data>::time(typename seg_data::vertex v) const
    {
        return data.fst->time(v);
    }

}
