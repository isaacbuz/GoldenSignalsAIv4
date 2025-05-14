import os
import pathlib


def generate_tree(startpath, max_depth=10, exclude_patterns=None):
    """
    Generate a tree-like representation of a directory structure.

    Args:
        startpath (str): Root directory path
        max_depth (int): Maximum depth of directory traversal
        exclude_patterns (list): List of patterns to exclude from the tree

    Returns:
        str: Formatted tree structure
    """
    if exclude_patterns is None:
        exclude_patterns = [
            ".git",
            ".github",
            "__pycache__",
            ".DS_Store",
            "*.pyc",
            ".pytest_cache",
            ".coverage",
            "node_modules",
            "dist",
            "build",
        ]

    def should_exclude(path):
        return any(
            pathlib.Path(path).match(pattern) or pattern in str(path)
            for pattern in exclude_patterns
        )

    def tree(directory, prefix="", depth=0):
        if depth > max_depth:
            return ""

        contents = sorted(os.listdir(directory))
        contents = [
            c for c in contents if not should_exclude(os.path.join(directory, c))
        ]

        tree_str = ""
        for i, item in enumerate(contents):
            path = os.path.join(directory, item)
            is_last = i == len(contents) - 1

            # Determine tree prefix
            current_prefix = prefix + ("└── " if is_last else "├── ")
            next_prefix = prefix + ("    " if is_last else "│   ")

            if os.path.isdir(path):
                tree_str += f"{current_prefix}{item}/\n"
                tree_str += tree(path, next_prefix, depth + 1)
            else:
                tree_str += f"{current_prefix}{item}\n"

        return tree_str

    print(f"Project Tree for: {startpath}")
    print(f"{'=' * 40}")
    full_tree = f"{os.path.basename(startpath)}/\n" + tree(startpath)
    return full_tree


def save_tree_to_file(tree_content, output_file):
    """Save the generated tree to a file."""
    with open(output_file, "w") as f:
        f.write(tree_content)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    tree_content = generate_tree(project_root)

    # Print to console
    print(tree_content)

    # Save to file
    output_file = os.path.join(project_root, "PROJECT_TREE.md")
    save_tree_to_file(tree_content, output_file)
    print(f"\nTree structure saved to {output_file}")


if __name__ == "__main__":
    main()
