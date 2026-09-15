import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { chooseSelectOption } from "../../../tests/test-utils";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import type { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "./team_dropdown";

const TEAMS = [
  { team_id: "team-1", team_alias: "Alpha Team" },
  { team_id: "team-2", team_alias: "Beta Team" },
] as unknown as Team[];

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: vi.fn(),
}));

const mockTeamsResult = (
  overrides: Partial<{
    pages: { teams: Team[] }[];
    fetchNextPage: () => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    isFetchNextPageError: boolean;
  }> = {},
) => {
  const { pages = [{ teams: TEAMS }], ...rest } = overrides;
  return {
    data: { pages },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isFetchNextPageError: false,
    isLoading: false,
    ...rest,
  } as unknown as ReturnType<typeof useInfiniteTeams>;
};

describe("TeamDropdown", () => {
  const mockUseInfiniteTeams = vi.mocked(useInfiniteTeams);

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult());
  });

  it("offers only teams accepted by the caller's permission filter", async () => {
    const user = userEvent.setup();
    render(<TeamDropdown filterTeam={(team) => team.team_id === "team-2"} />);
    await user.click(screen.getByRole("combobox"));
    expect(screen.queryByRole("option", { name: /Alpha Team/ })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Beta Team/ })).toBeInTheDocument();
  });

  it("loads past an empty eligible page so a permitted team on the next page can be selected", async () => {
    const user = userEvent.setup();
    const fetchNextPage = vi.fn();
    const onChange = vi.fn();
    const filterTeam = (team: Team) => team.team_id === "team-2";
    mockUseInfiniteTeams.mockReturnValue(
      mockTeamsResult({ pages: [{ teams: [TEAMS[0]] }], hasNextPage: true, fetchNextPage }),
    );
    const view = render(<TeamDropdown filterTeam={filterTeam} onChange={onChange} />);

    await waitFor(() => expect(fetchNextPage).toHaveBeenCalledOnce());

    mockUseInfiniteTeams.mockReturnValue(
      mockTeamsResult({ pages: [{ teams: [TEAMS[0]] }, { teams: [TEAMS[1]] }], fetchNextPage }),
    );
    view.rerender(<TeamDropdown filterTeam={filterTeam} onChange={onChange} />);
    await chooseSelectOption(user, screen.getByRole("combobox"), /^Beta Team/);

    expect(onChange).toHaveBeenCalledWith("team-2");
    expect(fetchNextPage).toHaveBeenCalledOnce();
  });

  it("fills a sparse filtered page and stops once a page of eligible teams is available", async () => {
    const fetchNextPage = vi.fn();
    const filterTeam = () => true;
    mockUseInfiniteTeams.mockReturnValue(
      mockTeamsResult({ pages: [{ teams: [TEAMS[0]] }], hasNextPage: true, fetchNextPage }),
    );
    const view = render(<TeamDropdown filterTeam={filterTeam} pageSize={2} />);

    await waitFor(() => expect(fetchNextPage).toHaveBeenCalledOnce());

    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult({ hasNextPage: true, fetchNextPage }));
    view.rerender(<TeamDropdown filterTeam={filterTeam} pageSize={2} />);

    expect(fetchNextPage).toHaveBeenCalledOnce();
  });

  it.each(["in-flight", "failed"])("does not repeat a page request while it is %s", (state) => {
    const fetchNextPage = vi.fn();
    const paginationState = {
      hasNextPage: true,
      fetchNextPage,
      isFetchingNextPage: state === "in-flight",
      isFetchNextPageError: state === "failed",
    };
    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult(paginationState));
    render(<TeamDropdown filterTeam={() => false} />);

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it("keeps unfiltered pagination driven by scrolling", () => {
    const fetchNextPage = vi.fn();
    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult({ hasNextPage: true, fetchNextPage }));
    render(<TeamDropdown />);

    expect(fetchNextPage).not.toHaveBeenCalled();
  });

  it("emits the picked team's id and full object", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onTeamSelect = vi.fn();
    render(<TeamDropdown onChange={onChange} onTeamSelect={onTeamSelect} />);

    await chooseSelectOption(user, screen.getByRole("combobox"), /^Beta Team/);

    expect(onChange).toHaveBeenCalledWith("team-2");
    expect(onTeamSelect).toHaveBeenCalledWith(TEAMS[1]);
  });

  it("emits null, never the empty string, when the selection is cleared", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onTeamSelect = vi.fn();
    render(<TeamDropdown value="team-1" onChange={onChange} onTeamSelect={onTeamSelect} />);

    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(onChange).toHaveBeenCalledWith(null);
    expect(onTeamSelect).toHaveBeenCalledWith(null);
  });
});
