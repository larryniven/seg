#include "scrf/scrf.h"
#include "scrf/lattice.h"
#include "ebt/ebt.h"
#include <fstream>

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "check-gold",
        "Check if the ground truth is in lattice",
        {
            {"lattice", "", true},
            {"gold", "", true}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    std::ifstream lattice_file { args.at("lattice") };
    std::ifstream gold_file { args.at("gold") };

    int miss = 0;
    int total = 0;

    while (1) {

        lattice::fst gold = scrf::load_gold(gold_file);
        lattice::fst lat = lattice::load_lattice(lattice_file);

        if (gold.edges().size() == 0 || lat.edges().size() == 0) {
            break;
        }

        auto time = [&](lattice::fst const& lat, int v) {
            return lat.data->vertices.at(v).time;
        };

        int u2 = lat.initial();

        for (int e = 0; e < gold.edges().size(); ++e) {
            bool found = false;

            for (auto& e2: lat.out_edges(u2)) {
                if (gold.output(e) == lat.output(e2)
                        && time(gold, gold.tail(e)) == time(lat, lat.tail(e2))
                        && time(gold, gold.head(e)) == time(lat, lat.head(e2))) {

                    u2 = lat.head(e2);
                    found = true;
                    break;
                }
            }

            if (!found) {
                std::cout << time(gold, gold.tail(e)) << " " << time(gold, gold.head(e)) << " " << gold.output(e) << " missed"<< std::endl;

                ++miss;
                break;
            }
        }

        ++total;

        std::cout << "miss: " << miss << " total: " << total << " miss rate: " << real(miss) / total << std::endl;
    }

    return 0;
}
