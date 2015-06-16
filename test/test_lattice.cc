#include <vector>
#include <utility>
#include "scrf/lattice.h"
#include "ebt/ebt.h"
#include <sstream>

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    {
        "test trivial lattice",
        []() {
            std::string trivial_lattice =
                "filename\n"
                "#\n"
                ".\n";
            std::istringstream iss { trivial_lattice };
            lattice::fst lat = lattice::load_lattice(iss);

            ebt::assert_equals(0, lat.edges().size());
            ebt::assert_equals(0, lat.vertices().size());
        }
    },
    {
        "test simple lattice",
        []() {
            std::string trivial_lattice =
                "filename\n"
                "0 time=0\n"
                "1 time=100000\n"
                "2 time=200000\n"
                "#\n"
                "0 1 weight=1.2,label=foo,graph_score=1.0\n"
                "1 2 weight=3.4,label=bar,graph_score=2.0\n"
                ".\n";
            std::istringstream iss { trivial_lattice };
            lattice::fst lat = lattice::load_lattice(iss);

            ebt::assert_equals(3, lat.vertices().size());
            ebt::assert_equals(2, lat.edges().size());
            ebt::assert_equals(0, lat.data->vertices.at(0).time);
            ebt::assert_equals(1, lat.data->vertices.at(1).time);
            ebt::assert_equals(2, lat.data->vertices.at(2).time);
            ebt::assert_equals(1.2, lat.weight(0));
            ebt::assert_equals(0, lat.tail(0));
            ebt::assert_equals(1, lat.head(0));
            ebt::assert_equals(1.0, lat.data->features.at(0)("graph_score"));
            ebt::assert_equals(3.4, lat.weight(1));
            ebt::assert_equals(1, lat.tail(1));
            ebt::assert_equals(2, lat.head(1));
            ebt::assert_equals(2.0, lat.data->features.at(1)("graph_score"));
        }
    },
};

int main()
{
    for (auto& p: tests) {
        p.second();
    }

    return 0;
}
