import { render, screen } from "@testing-library/react";
import AdminUserManagement from "../../components/admin/AdminUserManagement";

jest.mock("swr", () => () => ({
  data: [{ id: "1", name: "Test User" }],
  error: null,
  isLoading: false,
}));

describe("AdminUserManagement", () => {
  it("renders user data", () => {
    render(<AdminUserManagement />);
    expect(screen.getByText("Users")).toBeInTheDocument();
    expect(screen.getByText("Test User")).toBeInTheDocument();
  });
});
