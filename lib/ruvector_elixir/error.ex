defmodule RuvectorElixir.Error do
  @moduledoc """
  Exception raised by bang (`!`) functions in `RuvectorElixir` when a database
  or NIF operation fails.
  """

  @type reason :: :closed | :not_found | String.t() | atom()
  @type t :: %__MODULE__{message: String.t(), reason: reason()}

  defexception [:message, :reason]

  @impl true
  def exception(reason) when is_binary(reason) do
    %__MODULE__{message: reason, reason: reason}
  end

  def exception(:closed) do
    %__MODULE__{message: "Database handle is closed", reason: :closed}
  end

  def exception(:not_found) do
    %__MODULE__{message: "Vector entry not found", reason: :not_found}
  end

  def exception(reason) when is_atom(reason) do
    %__MODULE__{message: "Ruvector error: #{reason}", reason: reason}
  end
end
