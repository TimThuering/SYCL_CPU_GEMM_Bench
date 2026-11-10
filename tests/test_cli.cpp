// Command line parsing and validation.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "bench/cli.hpp"

namespace {

// parse_cli on an argument list (program name added)
bench::CliResult parse(std::vector<std::string> args) {
  args.insert(args.begin(), "gemm_bench");
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  return bench::parse_cli(static_cast<int>(argv.size()), argv.data());
}

TEST(Cli, Defaults) {
  const auto r = parse({"-s", "naive"});
  ASSERT_EQ(r.action, bench::CliAction::run);
  const auto& c = r.config;
  EXPECT_EQ(c.stage, "naive");
  EXPECT_EQ(c.problem.m, 1000u);
  EXPECT_EQ(c.problem.n, 1000u);
  EXPECT_EQ(c.problem.k, 1000u);
  EXPECT_EQ(c.problem.lda, 1000u);
  EXPECT_EQ(c.seed, 123u);
  EXPECT_EQ(c.warmup, 2u);
  EXPECT_EQ(c.reps, 10u);
  EXPECT_TRUE(c.verify);
  EXPECT_TRUE(c.output.empty());
  EXPECT_FALSE(c.quiet);
}

TEST(Cli, AllOptions) {
  const auto r = parse({"--stage", "regblock", "--size", "100", "-n", "77", "-k", "33", "--ld-pad", "5",
                        "--seed", "7", "--warmup", "0", "--reps", "3", "--no-verify", "-o", "out.json",
                        "--label", "a b", "-q"});
  const auto& c = r.config;
  EXPECT_EQ(c.stage, "regblock");
  EXPECT_EQ(c.problem.m, 100u);
  EXPECT_EQ(c.problem.n, 77u);
  EXPECT_EQ(c.problem.k, 33u);
  EXPECT_EQ(c.problem.lda, 38u);  // k + pad
  EXPECT_EQ(c.problem.ldb, 82u);  // n + pad
  EXPECT_EQ(c.problem.ldc, 82u);
  EXPECT_EQ(c.seed, 7u);
  EXPECT_EQ(c.warmup, 0u);
  EXPECT_EQ(c.reps, 3u);
  EXPECT_FALSE(c.verify);
  EXPECT_EQ(c.output, "out.json");
  EXPECT_EQ(c.label, "a b");
  EXPECT_TRUE(c.quiet);
  EXPECT_NE(c.command_line.find("'a b'"), std::string::npos);
}

TEST(Cli, InfoActions) {
  EXPECT_EQ(parse({"--help"}).action, bench::CliAction::help);
  EXPECT_EQ(parse({"--version"}).action, bench::CliAction::version);
  EXPECT_EQ(parse({"--list-stages"}).action, bench::CliAction::list_stages);
  EXPECT_EQ(parse({"--list-devices"}).action, bench::CliAction::list_devices);
}

TEST(Cli, InvalidInput) {
  EXPECT_THROW(parse({}), bench::UsageError);                                // missing stage
  EXPECT_THROW(parse({"-s", "foo"}), bench::UsageError);                     // unknown stage
  EXPECT_THROW(parse({"-s", "naive", "--size", "0"}), bench::UsageError);    // zero size
  EXPECT_THROW(parse({"-s", "naive", "-m", "-5"}), bench::UsageError);       // negative
  EXPECT_THROW(parse({"-s", "naive", "--size", "1e3"}), bench::UsageError);  // not an integer
  EXPECT_THROW(parse({"-s", "naive", "--size", "1000001"}), bench::UsageError);  // too large
  EXPECT_THROW(parse({"-s", "naive", "--reps", "0"}), bench::UsageError);
  EXPECT_THROW(parse({"-s", "naive", "--bogus"}), bench::UsageError);        // unknown option
  EXPECT_THROW(parse({"-s", "naive", "extra"}), bench::UsageError);          // positional argument
}

} // namespace
