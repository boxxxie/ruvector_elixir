defmodule RuvectorElixir.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/ruvnet/ruvector"

  def project do
    [
      app: :ruvector_elixir,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      docs: docs(),
      deps: deps(),
      source_url: @source_url
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
        "mix.exs",
        "README.md",
        "LICENSE"
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
      {:rustler, "~> 0.36.0"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
