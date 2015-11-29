#include "scrf/fst.h"
#include "scrf/lattice.h"

lattice::fst make_fst1()
{
    lattice::fst_data data;

    lattice::add_vertex(data, 0, 0);
    lattice::add_vertex(data, 1, 1);
    lattice::add_vertex(data, 2, 2);

    lattice::add_edge(data, 0, "a", 0, 1, 1);
    lattice::add_edge(data, 1, "b", 0, 1, 2);
    lattice::add_edge(data, 2, "c", 1, 2, 3);
    lattice::add_edge(data, 3, "d", 1, 2, 4);

    data.initials.push_back(0);
    data.finals.push_back(2);

    lattice::fst fst;

    fst.data = std::make_shared<lattice::fst_data>(data);

    return fst;
}

lattice::fst make_fst2()
{
    lattice::fst_data data;

    lattice::add_vertex(data, 0, 0);
    lattice::add_vertex(data, 1, 1);
    lattice::add_vertex(data, 2, 2);

    lattice::add_edge(data, 0, "a", 0, 2, 1);
    lattice::add_edge(data, 1, "b", 0, 1, 2);
    lattice::add_edge(data, 2, "c", 1, 2, 3);

    data.initials.push_back(0);
    data.finals.push_back(2);

    lattice::fst fst;

    fst.data = std::make_shared<lattice::fst_data>(data);

    return fst;
}

lattice::fst make_fst3()
{
    lattice::fst_data data;

    lattice::add_vertex(data, 0, 0);
    lattice::add_vertex(data, 1, 1);
    lattice::add_vertex(data, 2, 2);
    lattice::add_vertex(data, 3, 3);

    lattice::add_edge(data, 0, "a", 0, 2, 1);
    lattice::add_edge(data, 1, "b", 1, 3, 2);
    lattice::add_edge(data, 2, "c", 0, 1, 3);
    lattice::add_edge(data, 3, "d", 1, 2, 4);
    lattice::add_edge(data, 4, "e", 2, 3, 5);

    data.initials.push_back(0);
    data.finals.push_back(3);

    lattice::fst fst;

    fst.data = std::make_shared<lattice::fst_data>(data);

    return fst;
}

std::vector<std::function<void(void)>> tests = {
    []() {
        lattice::fst g = make_fst1();

        std::vector<int> topo_order = fst::topo_order(g);

        fst::lazy_k_best<lattice::fst> kbest;

        kbest.one_best(g, topo_order);

        fst::path<lattice::fst> path = kbest.backtrack(g, 2, 0);

        auto edges = path.edges();
        ebt::assert_equals("b", path.output(edges.at(0)));
        ebt::assert_equals("d", path.output(edges.at(1)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 2, 1);
        edges = path.edges();
        ebt::assert_equals("b", path.output(edges.at(0)));
        ebt::assert_equals("c", path.output(edges.at(1)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 2, 2);
        edges = path.edges();
        ebt::assert_equals("a", path.output(edges.at(0)));
        ebt::assert_equals("d", path.output(edges.at(1)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 2, 3);
        edges = path.edges();
        ebt::assert_equals("a", path.output(edges.at(0)));
        ebt::assert_equals("c", path.output(edges.at(1)));
    },

    []() {
        lattice::fst g = make_fst2();

        std::vector<int> topo_order = fst::topo_order(g);

        fst::lazy_k_best<lattice::fst> kbest;

        kbest.one_best(g, topo_order);

        fst::path<lattice::fst> path = kbest.backtrack(g, 2, 0);

        auto edges = path.edges();
        ebt::assert_equals("b", path.output(edges.at(0)));
        ebt::assert_equals("c", path.output(edges.at(1)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 2, 1);
        edges = path.edges();
        ebt::assert_equals("a", path.output(edges.at(0)));
    },

    []() {
        lattice::fst g = make_fst3();

        std::vector<int> topo_order = fst::topo_order(g);

        fst::lazy_k_best<lattice::fst> kbest;

        kbest.one_best(g, topo_order);

        fst::path<lattice::fst> path = kbest.backtrack(g, 3, 0);

        auto edges = path.edges();
        ebt::assert_equals("c", path.output(edges.at(0)));
        ebt::assert_equals("d", path.output(edges.at(1)));
        ebt::assert_equals("e", path.output(edges.at(2)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 3, 1);
        edges = path.edges();
        ebt::assert_equals("a", path.output(edges.at(0)));
        ebt::assert_equals("e", path.output(edges.at(1)));

        kbest.update(g, path);
        path = kbest.backtrack(g, 3, 2);
        edges = path.edges();
        ebt::assert_equals("c", path.output(edges.at(0)));
        ebt::assert_equals("b", path.output(edges.at(1)));
    }
};

int main()
{
    for (auto& t: tests) {
        t();
    }

    return 0;
}
