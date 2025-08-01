import React, { useState, useMemo, useCallback, ReactNode } from 'react';
import clsx from 'clsx';
import { ChevronUpIcon, ChevronDownIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { logger as frontendLogger } from '../../services/logging/logger';
import { Input } from '../Input/Input';
import { Button } from '../Core/Button/Button';
import styles from './Table.module.css';

export interface Column<T> {
    /** Unique identifier for the column */
    id: string;
    /** Display header for the column */
    header: string;
    /** Accessor function or property key */
    accessor: keyof T | ((row: T) => any);
    /** Custom cell renderer */
    cell?: (value: any, row: T) => ReactNode;
    /** Whether the column is sortable */
    sortable?: boolean;
    /** Custom sort function */
    sortFn?: (a: T, b: T) => number;
    /** Column width */
    width?: string | number;
    /** Alignment */
    align?: 'left' | 'center' | 'right';
}

export interface TableProps<T> {
    /** Table data */
    data: T[];
    /** Column definitions */
    columns: Column<T>[];
    /** Whether to show row selection checkboxes */
    selectable?: boolean;
    /** Selected row IDs */
    selectedRows?: Set<string | number>;
    /** Callback when row selection changes */
    onSelectionChange?: (selectedRows: Set<string | number>) => void;
    /** Row key accessor */
    getRowId?: (row: T) => string | number;
    /** Whether to enable sorting */
    sortable?: boolean;
    /** Whether to enable filtering */
    filterable?: boolean;
    /** Filter placeholder text */
    filterPlaceholder?: string;
    /** Whether to enable pagination */
    paginated?: boolean;
    /** Page size options */
    pageSizeOptions?: number[];
    /** Default page size */
    defaultPageSize?: number;
    /** Loading state */
    loading?: boolean;
    /** Empty state message */
    emptyMessage?: string;
    /** Row click handler */
    onRowClick?: (row: T) => void;
    /** Whether rows are clickable */
    clickableRows?: boolean;
    /** Striped rows */
    striped?: boolean;
    /** Bordered table */
    bordered?: boolean;
    /** Compact mode */
    compact?: boolean;
    /** Sticky header */
    stickyHeader?: boolean;
    /** Maximum height for scrollable table */
    maxHeight?: string | number;
    /** Additional CSS classes */
    className?: string;
    /** Test ID */
    'data-testid'?: string;
}

type SortDirection = 'asc' | 'desc' | null;

export function Table<T extends Record<string, any>>({
    data,
    columns,
    selectable = false,
    selectedRows = new Set(),
    onSelectionChange,
    getRowId = (row) => row.id,
    sortable = true,
    filterable = false,
    filterPlaceholder = 'Search...',
    paginated = false,
    pageSizeOptions = [10, 25, 50, 100],
    defaultPageSize = 10,
    loading = false,
    emptyMessage = 'No data available',
    onRowClick,
    clickableRows = false,
    striped = false,
    bordered = false,
    compact = false,
    stickyHeader = false,
    maxHeight,
    className,
    'data-testid': testId,
}: TableProps<T>) {
    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<SortDirection>(null);
    const [filterValue, setFilterValue] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const [pageSize, setPageSize] = useState(defaultPageSize);

    // Get cell value
    const getCellValue = useCallback((row: T, column: Column<T>) => {
        if (typeof column.accessor === 'function') {
            return column.accessor(row);
        }
        return row[column.accessor];
    }, []);

    // Filter data
    const filteredData = useMemo(() => {
        if (!filterable || !filterValue) return data;

        return data.filter((row) => {
            return columns.some((column) => {
                const value = getCellValue(row, column);
                return String(value).toLowerCase().includes(filterValue.toLowerCase());
            });
        });
    }, [data, filterValue, filterable, columns, getCellValue]);

    // Sort data
    const sortedData = useMemo(() => {
        if (!sortColumn || !sortDirection) return filteredData;

        const column = columns.find((col) => col.id === sortColumn);
        if (!column) return filteredData;

        return [...filteredData].sort((a, b) => {
            if (column.sortFn) {
                return sortDirection === 'asc' ? column.sortFn(a, b) : column.sortFn(b, a);
            }

            const aValue = getCellValue(a, column);
            const bValue = getCellValue(b, column);

            if (aValue === bValue) return 0;

            const result = aValue < bValue ? -1 : 1;
            return sortDirection === 'asc' ? result : -result;
        });
    }, [filteredData, sortColumn, sortDirection, columns, getCellValue]);

    // Paginate data
    const paginatedData = useMemo(() => {
        if (!paginated) return sortedData;

        const startIndex = (currentPage - 1) * pageSize;
        return sortedData.slice(startIndex, startIndex + pageSize);
    }, [sortedData, paginated, currentPage, pageSize]);

    const totalPages = Math.ceil(filteredData.length / pageSize);

    // Handle sorting
    const handleSort = (columnId: string) => {
        if (!sortable) return;

        if (sortColumn === columnId) {
            setSortDirection((prev) => {
                if (prev === 'asc') return 'desc';
                if (prev === 'desc') return null;
                return 'asc';
            });
        } else {
            setSortColumn(columnId);
            setSortDirection('asc');
        }

        frontendLogger.debug('Table sorted', { columnId, sortDirection });
    };

    // Handle row selection
    const handleSelectAll = () => {
        if (!onSelectionChange) return;

        const allRowIds = paginatedData.map((row) => getRowId(row));
        const newSelection = new Set(selectedRows);

        const allSelected = allRowIds.every((id) => selectedRows.has(id));

        if (allSelected) {
            allRowIds.forEach((id) => newSelection.delete(id));
        } else {
            allRowIds.forEach((id) => newSelection.add(id));
        }

        onSelectionChange(newSelection);
    };

    const handleSelectRow = (rowId: string | number) => {
        if (!onSelectionChange) return;

        const newSelection = new Set(selectedRows);
        if (newSelection.has(rowId)) {
            newSelection.delete(rowId);
        } else {
            newSelection.add(rowId);
        }

        onSelectionChange(newSelection);
    };

    // Handle pagination
    const handlePageChange = (page: number) => {
        setCurrentPage(page);
        frontendLogger.debug('Table page changed', { page });
    };

    const handlePageSizeChange = (newSize: number) => {
        setPageSize(newSize);
        setCurrentPage(1);
        frontendLogger.debug('Table page size changed', { pageSize: newSize });
    };

    // Table classes
    const tableClasses = clsx(
        styles.table,
        {
            [styles.striped]: striped,
            [styles.bordered]: bordered,
            [styles.compact]: compact,
            [styles.stickyHeader]: stickyHeader,
            [styles.clickableRows]: clickableRows,
        },
        className
    );

    const containerClasses = clsx(
        styles.container,
        {
            [styles.scrollable]: maxHeight,
        }
    );

    const containerStyle = maxHeight ? { maxHeight } : undefined;

    // Loading state
    if (loading) {
        return (
            <div className={styles.loadingContainer} data-testid={`${testId}-loading`}>
                <div className={styles.spinner} />
                <p>Loading...</p>
            </div>
        );
    }

    // Empty state
    if (paginatedData.length === 0) {
        return (
            <div className={styles.emptyContainer} data-testid={`${testId}-empty`}>
                <p>{emptyMessage}</p>
            </div>
        );
    }

    return (
        <div className={styles.wrapper} data-testid={testId}>
            {/* Filter */}
            {filterable && (
                <div className={styles.filterContainer}>
                    <Input
                        placeholder={filterPlaceholder}
                        value={filterValue}
                        onChange={(e) => setFilterValue(e.target.value)}
                        startIcon={<MagnifyingGlassIcon className="w-5 h-5" />}
                        size="small"
                        data-testid={`${testId}-filter`}
                    />
                </div>
            )}

            {/* Table */}
            <div className={containerClasses} style={containerStyle}>
                <table className={tableClasses}>
                    <thead>
                        <tr>
                            {selectable && (
                                <th className={styles.checkboxCell}>
                                    <input
                                        type="checkbox"
                                        checked={paginatedData.every((row) => selectedRows.has(getRowId(row)))}
                                        onChange={handleSelectAll}
                                        aria-label="Select all rows"
                                    />
                                </th>
                            )}
                            {columns.map((column) => (
                                <th
                                    key={column.id}
                                    className={clsx(
                                        styles.headerCell,
                                        styles[`align-${column.align || 'left'}`],
                                        {
                                            [styles.sortable]: sortable && column.sortable !== false,
                                        }
                                    )}
                                    style={{ width: column.width }}
                                    onClick={() => column.sortable !== false && handleSort(column.id)}
                                >
                                    <div className={styles.headerContent}>
                                        <span>{column.header}</span>
                                        {sortable && column.sortable !== false && (
                                            <span className={styles.sortIcon}>
                                                {sortColumn === column.id && sortDirection === 'asc' && (
                                                    <ChevronUpIcon className="w-4 h-4" />
                                                )}
                                                {sortColumn === column.id && sortDirection === 'desc' && (
                                                    <ChevronDownIcon className="w-4 h-4" />
                                                )}
                                            </span>
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedData.map((row) => {
                            const rowId = getRowId(row);
                            const isSelected = selectedRows.has(rowId);

                            return (
                                <tr
                                    key={rowId}
                                    className={clsx({
                                        [styles.selected]: isSelected,
                                        [styles.clickable]: clickableRows,
                                    })}
                                    onClick={() => onRowClick?.(row)}
                                >
                                    {selectable && (
                                        <td className={styles.checkboxCell}>
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => handleSelectRow(rowId)}
                                                onClick={(e) => e.stopPropagation()}
                                                aria-label={`Select row ${rowId}`}
                                            />
                                        </td>
                                    )}
                                    {columns.map((column) => {
                                        const value = getCellValue(row, column);
                                        const cellContent = column.cell ? column.cell(value, row) : value;

                                        return (
                                            <td
                                                key={column.id}
                                                className={clsx(
                                                    styles.cell,
                                                    styles[`align-${column.align || 'left'}`]
                                                )}
                                            >
                                                {cellContent}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            {paginated && totalPages > 1 && (
                <div className={styles.pagination}>
                    <div className={styles.pageInfo}>
                        Showing {((currentPage - 1) * pageSize) + 1} to{' '}
                        {Math.min(currentPage * pageSize, filteredData.length)} of{' '}
                        {filteredData.length} entries
                    </div>

                    <div className={styles.pageControls}>
                        <Button
                            variant="ghost"
                            size="small"
                            onClick={() => handlePageChange(currentPage - 1)}
                            disabled={currentPage === 1}
                        >
                            Previous
                        </Button>

                        <div className={styles.pageNumbers}>
                            {Array.from({ length: totalPages }, (_, i) => i + 1)
                                .filter((page) => {
                                    return page === 1 ||
                                        page === totalPages ||
                                        Math.abs(page - currentPage) <= 2;
                                })
                                .map((page, index, array) => (
                                    <React.Fragment key={page}>
                                        {index > 0 && array[index - 1] !== page - 1 && (
                                            <span className={styles.ellipsis}>...</span>
                                        )}
                                        <button
                                            className={clsx(styles.pageButton, {
                                                [styles.active]: page === currentPage,
                                            })}
                                            onClick={() => handlePageChange(page)}
                                        >
                                            {page}
                                        </button>
                                    </React.Fragment>
                                ))}
                        </div>

                        <Button
                            variant="ghost"
                            size="small"
                            onClick={() => handlePageChange(currentPage + 1)}
                            disabled={currentPage === totalPages}
                        >
                            Next
                        </Button>
                    </div>

                    <div className={styles.pageSizeSelector}>
                        <label htmlFor="page-size">Show:</label>
                        <select
                            id="page-size"
                            value={pageSize}
                            onChange={(e) => handlePageSizeChange(Number(e.target.value))}
                            className={styles.pageSizeSelect}
                        >
                            {pageSizeOptions.map((size) => (
                                <option key={size} value={size}>
                                    {size}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            )}
        </div>
    );
}
