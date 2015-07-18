#include "scrf/scrf.h"
#include "ebt/ebt.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    {
        "test fst composition", []() {
            auto lat = std::make_shared<lattice::fst>(scrf::make_segmentation_lattice(2, 2));

            auto lm = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));
            lm = scrf::erase_input(lm);

            fst::composed_fst<lattice::fst, lm::fst> fst;
            fst.fst1 = lat;
            fst.fst2 = lm;

            auto edges = fst.edges();

            std::ostringstream oss;

            for (auto& e: fst.edges()) {
                oss << fst.tail(e) << " " << fst.head(e) << " " << fst.output(e) << std::endl;
            }

            ebt::assert_equals(
                "(0, 0) (1, 0) <s>\n"
                "(0, 0) (1, 0) </s>\n"
                "(0, 0) (2, 0) <s>\n"
                "(0, 0) (2, 0) </s>\n"
                "(1, 0) (2, 0) <s>\n"
                "(1, 0) (2, 0) </s>\n"
                , oss.str());
        }
    },
    {
        "test scrf score", []() {
            std::vector<std::vector<real>> inputs {
                {2, 1},
                {1, 2}
            };

            auto lat = std::make_shared<lattice::fst>(scrf::make_segmentation_lattice(2, 2));

            auto lm = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));
            lm = scrf::erase_input(lm);

            fst::composed_fst<lattice::fst, lm::fst> fst;
            fst.fst1 = lat;
            fst.fst2 = lm;

            scrf::param_t param { {{"<s>", {1, 0}}, {"</s>", {0, 1}}} };
            scrf::feature::frame_samples feat_func { inputs, 1 };
            scrf::scrf_t scrf { std::make_shared<decltype(fst)>(fst) };
            scrf.weight_func =  std::make_shared<scrf::linear_score>(
                scrf::linear_score { param, feat_func });

            std::ostringstream oss;

            for (auto& e: scrf.edges()) {
                oss << scrf.tail(e) << " " << scrf.head(e)
                    << " " << scrf.output(e) << " " << scrf.weight(e) << std::endl;
            }

            ebt::assert_equals(
                "(0, 0) (1, 0) <s> 2\n"
                "(0, 0) (1, 0) </s> 1\n"
                "(0, 0) (2, 0) <s> 1\n"
                "(0, 0) (2, 0) </s> 2\n"
                "(1, 0) (2, 0) <s> 1\n"
                "(1, 0) (2, 0) </s> 2\n"
                , oss.str());

        }
    },
    {
        "test scrf one best 2 frames", []() {
            std::vector<std::vector<real>> inputs {
                {2, 1},
                {1, 2}
            };

            auto lat = std::make_shared<lattice::fst>(scrf::make_segmentation_lattice(2, 2));

            auto lm = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));
            lm = scrf::erase_input(lm);

            fst::composed_fst<lattice::fst, lm::fst> fst;
            fst.fst1 = lat;
            fst.fst2 = lm;

            scrf::param_t param { {{"<s>", {1, 0}}, {"</s>", {0, 1}}} };
            scrf::feature::frame_samples feat_func { inputs, 1 };
            scrf::scrf_t scrf { std::make_shared<decltype(fst)>(fst) };
            scrf.weight_func = std::make_shared<scrf::linear_score>(
                scrf::linear_score { param, feat_func });

            fst::one_best<scrf::scrf_t> one_best;

            for (auto& i: scrf.initials()) {
                one_best.extra[i] = fst::one_best<scrf::scrf_t>::extra_data {
                    std::make_tuple(-1, -1), 0 };
            }

            one_best.merge(scrf, scrf.vertices());

            std::ostringstream oss;

            for (auto& p: one_best.extra) {
                oss << p.first << " " << p.second.value <<
                    " " << scrf.tail(p.second.pi) << std::endl;
            }

            ebt::assert_equals(
                "(2, 0) 4 (1, 0)\n"
                "(1, 0) 2 (0, 0)\n"
                "(0, 0) 0 (0, 0)\n"
                , oss.str());

        }
    },
    {
        "test scrf one best 3 frames", []() {
            std::vector<std::vector<real>> inputs {
                {2, 1},
                {2, 1},
                {1, 2},
            };

            auto lat = std::make_shared<lattice::fst>(scrf::make_segmentation_lattice(3, 2));

            auto lm = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));
            lm = scrf::erase_input(lm);

            fst::composed_fst<lattice::fst, lm::fst> fst;
            fst.fst1 = lat;
            fst.fst2 = lm;

            scrf::param_t param { {{"<s>", {1, 0}}, {"</s>", {0, 1}}} };
            scrf::feature::frame_samples feat_func { inputs, 1 };
            scrf::scrf_t scrf { std::make_shared<decltype(fst)>(fst) };
            scrf.weight_func = std::make_shared<scrf::linear_score>(
                scrf::linear_score { param, feat_func });

            fst::one_best<scrf::scrf_t> one_best;

            for (auto& i: scrf.initials()) {
                one_best.extra[i] = fst::one_best<scrf::scrf_t>::extra_data {
                    std::make_tuple(-1, -1), 0 };
            }

            one_best.merge(scrf, scrf.vertices());

            std::ostringstream oss;

            for (auto& p: one_best.extra) {
                oss << p.first << " " << p.second.value <<
                    " " << scrf.tail(p.second.pi) << std::endl;
            }

            ebt::assert_equals(
                "(3, 0) 6 (2, 0)\n"
                "(2, 0) 4 (1, 0)\n"
                "(1, 0) 2 (0, 0)\n"
                "(0, 0) 0 (0, 0)\n"
                , oss.str());

        }
    },
    {
        "test overlap cost", []() {
            std::string gold_str =
                "foo\n"
                "0 100000 <s>\n"
                "100000 300000 </s>\n"
                ".";

            std::istringstream iss { gold_str };
            lattice::fst gold_lat = scrf::load_gold(iss);

            auto lm1 = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));

            fst::composed_fst<lattice::fst, lm::fst> gold_fst;
            gold_fst.fst1 = std::make_shared<lattice::fst>(gold_lat);
            gold_fst.fst2 = lm1;

            scrf::scrf_t gold { std::make_shared<decltype(gold_fst)>(gold_fst) };
            gold.weight_func = std::make_shared<scrf::backoff_cost>(scrf::backoff_cost{});

            auto gold_path = scrf::shortest_path(gold, gold.vertices());

            auto lm2 = std::make_shared<lm::fst>(lm::load_arpa_lm("lm.2"));
            lm2 = scrf::erase_input(lm2);

            auto lat = std::make_shared<lattice::fst>(scrf::make_segmentation_lattice(3, 2));

            fst::composed_fst<lattice::fst, lm::fst> fst;
            fst.fst1 = lat;
            fst.fst2 = lm2;

            scrf::scrf_t scrf { std::make_shared<decltype(fst)>(fst) };
            scrf.weight_func = std::make_shared<scrf::overlap_cost>(
                scrf::overlap_cost { gold_path });

            std::ostringstream oss;

            for (auto& e: scrf.edges()) {
                oss << scrf.tail(e) << " " << scrf.head(e) << " " << scrf.output(e)
                    << " " << scrf.weight(e) << std::endl;
            }

            ebt::assert_equals(
                "(0, 0) (1, 0) <s> 0\n"
                "(0, 0) (1, 0) </s> 1\n"
                "(0, 0) (2, 0) <s> 1\n"
                "(0, 0) (2, 0) </s> 2\n"
                "(1, 0) (2, 0) <s> 2\n"
                "(1, 0) (2, 0) </s> 1\n"
                "(1, 0) (3, 0) <s> 2\n"
                "(1, 0) (3, 0) </s> 0\n"
                "(2, 0) (3, 0) <s> 2\n"
                "(2, 0) (3, 0) </s> 1\n"
                , oss.str());

        }
    }
};

int main()
{
    for (auto& p: tests) {
        std::cout << p.first << std::endl;
        p.second();
    }

    return 0;
}
