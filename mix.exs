defmodule RuvectorElixir.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/ruvnet/ruvector"

  def project do
    [
      app: :ruvector_elixir,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      docs: docs(),
      deps: deps(),
      aliases: aliases(),
      source_url: @source_url
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description do
    "High-performance Elixir bindings for ruvector, an ultra-fast embedded vector database and GraphRAG engine written in Rust."
  end

  defp package do
    [
      name: :ruvector_elixir,
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "ruvector" => @source_url
      },
      files: [
        "lib",
        "native/ruvector_nif/src",
        "native/ruvector_nif/Cargo.toml",
        "native/ruvector_nif/Cargo.lock",
        "native/ruvector_nif/README.md",
        "usage-rules.md",
        "mix.exs",
        ".formatter.exs",
        "README.md",
        "LICENSE"
      ]
    ]
  end

  defp aliases do
    [
      precommit: [
        "format --check-formatted",
        "cmd cargo fmt --manifest-path native/ruvector_nif/Cargo.toml --check",
        "cmd cargo clippy --manifest-path native/ruvector_nif/Cargo.toml -- -D warnings",
        "compile --force --warnings-as-errors",
        "test"
      ]
    ]
  end

  defp docs do
    [
      main: "RuvectorElixir",
      source_ref: "v#{@version}",
      source_url: @source_url,
      extras: ["README.md", "LICENSE"]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.38.0", runtime: false},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
