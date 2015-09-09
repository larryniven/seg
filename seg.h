#ifndef SEG_H
#define SEG_H

namespace seg {

    template <class base_fst, class weight_func_t, class feature_func_t>
    struct fst {
        using fst_type = base_fst;
        using vertex_type = typename base_fst::vertex_type;
        using edge_type = typename base_fst::edge_type;

        std::shared_ptr<fst_type> fst;
        std::shared_ptr<weight_func_t> weight_func;
        std::shared_ptr<feature_func_t> feature_func;

        std::vector<vertex_type> vertices() const
        {
            return fst->edges();
        }

        std::vector<edge_type> edges() const
        {
            return fst->edges();
        }

        vertex_type head(edge_type const& e) const
        {
            return fst->head(e);
        }

        vertex_type tail(edge_type const& e) const
        {
            return fst->tail(e);
        }

        std::vector<edge_type> in_edges(vertex_type const& v) const
        {
            return fst->in_edges(v);
        }

        std::vector<edge_type> out_edges(vertex_type const& v) const
        {
            return fst->out_edges(v);
        }

        real weight(edge_type const& e) const
        {
            return (*weight_func)(*fst, e);
        }

        std::string input(edge_type const& e) const
        {
            return fst->input(e);
        }

        std::string output(edge_type const& e) const
        {
            return fst->output(e);
        }

        std::vector<vertex_type> initials() const
        {
            return fst->initials();
        }

        std::vector<vertex_type> finals() const
        {
            return fst->finals();
        }

    };

}

#endif
