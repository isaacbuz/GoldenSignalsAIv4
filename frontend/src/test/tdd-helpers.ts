/**
 * TDD (Test-Driven Development) Helpers
 * Utilities to support Test-Driven Development methodology
 */

import React from 'react'
import { vi, expect, describe, it, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from './test-utils'

export interface TDDTestCase<T = any> {
    name: string
    given: () => T | Promise<T>
    when: (context: T) => void | Promise<void>
    then: (context: T) => void | Promise<void>
}

export interface BehaviorSpec {
    feature: string
    scenario: string
    given: string[]
    when: string[]
    then: string[]
}

/**
 * TDD Test Suite Builder
 * Provides a fluent API for building test-driven test suites
 */
export class TDDTestSuite<T = any> {
    private testCases: TDDTestCase<T>[] = []
    private setupFn?: () => T | Promise<T>
    private teardownFn?: (context: T) => void | Promise<void>
    private suiteName: string

    constructor(suiteName: string) {
        this.suiteName = suiteName
    }

    setup(setupFn: () => T | Promise<T>): this {
        this.setupFn = setupFn
        return this
    }

    teardown(teardownFn: (context: T) => void | Promise<void>): this {
        this.teardownFn = teardownFn
        return this
    }

    addTest(testCase: TDDTestCase<T>): this {
        this.testCases.push(testCase)
        return this
    }

    test(name: string): TDDTestBuilder<T> {
        return new TDDTestBuilder<T>(name, this)
    }

    run(): void {
        describe(this.suiteName, () => {
            let context: T

            beforeEach(async () => {
                if (this.setupFn) {
                    context = await this.setupFn()
                }
            })

            afterEach(async () => {
                if (this.teardownFn && context) {
                    await this.teardownFn(context)
                }
            })

            this.testCases.forEach(testCase => {
                it(testCase.name, async () => {
                    const testContext = await testCase.given()
                    await testCase.when(testContext)
                    await testCase.then(testContext)
                })
            })
        })
    }
}

export class TDDTestBuilder<T> {
    private givenFn?: () => T | Promise<T>
    private whenFn?: (context: T) => void | Promise<void>
    private thenFn?: (context: T) => void | Promise<void>

    constructor(
        private name: string,
        private suite: TDDTestSuite<T>
    ) { }

    given(givenFn: () => T | Promise<T>): this {
        this.givenFn = givenFn
        return this
    }

    when(whenFn: (context: T) => void | Promise<void>): this {
        this.whenFn = whenFn
        return this
    }

    then(thenFn: (context: T) => void | Promise<void>): this {
        this.thenFn = thenFn
        return this
    }

    build(): TDDTestSuite<T> {
        if (!this.givenFn || !this.whenFn || !this.thenFn) {
            throw new Error('Test case must have given, when, and then functions')
        }

        this.suite.addTest({
            name: this.name,
            given: this.givenFn,
            when: this.whenFn,
            then: this.thenFn
        })

        return this.suite
    }
}

/**
 * BDD-style test helpers
 */
export class BehaviorTestSuite {
    private specs: BehaviorSpec[] = []

    feature(name: string): FeatureBuilder {
        return new FeatureBuilder(name, this)
    }

    addSpec(spec: BehaviorSpec): void {
        this.specs.push(spec)
    }

    run(): void {
        this.specs.forEach(spec => {
            describe(`Feature: ${spec.feature}`, () => {
                describe(`Scenario: ${spec.scenario}`, () => {
                    it('should behave as expected', async () => {
                        // This is a simplified BDD runner
                        // In a real implementation, you'd parse and execute the steps
                        console.log('Given:', spec.given.join(', '))
                        console.log('When:', spec.when.join(', '))
                        console.log('Then:', spec.then.join(', '))
                    })
                })
            })
        })
    }
}

export class FeatureBuilder {
    constructor(
        private featureName: string,
        private suite: BehaviorTestSuite
    ) { }

    scenario(name: string): ScenarioBuilder {
        return new ScenarioBuilder(this.featureName, name, this.suite)
    }
}

export class ScenarioBuilder {
    private givenSteps: string[] = []
    private whenSteps: string[] = []
    private thenSteps: string[] = []

    constructor(
        private featureName: string,
        private scenarioName: string,
        private suite: BehaviorTestSuite
    ) { }

    given(step: string): this {
        this.givenSteps.push(step)
        return this
    }

    when(step: string): this {
        this.whenSteps.push(step)
        return this
    }

    then(step: string): this {
        this.thenSteps.push(step)
        return this
    }

    build(): BehaviorTestSuite {
        this.suite.addSpec({
            feature: this.featureName,
            scenario: this.scenarioName,
            given: this.givenSteps,
            when: this.whenSteps,
            then: this.thenSteps
        })

        return this.suite
    }
}

/**
 * Component Testing Helpers for TDD
 */
export class ComponentTDDHelper {
    static createTestSuite(componentName: string) {
        return new TDDTestSuite(`${componentName} Component`)
    }

    static async renderComponent(Component: React.ComponentType<any>, props: any = {}) {
        return renderWithProviders(React.createElement(Component, props))
    }

    static async userInteraction() {
        return userEvent.setup()
    }

    static expectElementToExist(selector: string) {
        const element = screen.queryByTestId(selector) || screen.queryByRole('button', { name: selector })
        expect(element).toBeInTheDocument()
        return element
    }

    static expectElementNotToExist(selector: string) {
        const element = screen.queryByTestId(selector) || screen.queryByRole('button', { name: selector })
        expect(element).not.toBeInTheDocument()
    }

    static async expectElementToAppear(selector: string, timeout = 1000) {
        await waitFor(() => {
            const element = screen.queryByTestId(selector) || screen.queryByRole('button', { name: selector })
            expect(element).toBeInTheDocument()
        }, { timeout })
    }

    static async expectTextToAppear(text: string, timeout = 1000) {
        await waitFor(() => {
            expect(screen.getByText(text)).toBeInTheDocument()
        }, { timeout })
    }

    static createMockProps<T>(overrides: Partial<T> = {}): T {
        return {
            ...overrides
        } as T
    }
}

/**
 * API Testing Helpers for TDD
 */
export class APIMockHelper {
    private mocks: Map<string, any> = new Map()

    mockApiCall(endpoint: string, response: any, options: { delay?: number, error?: boolean } = {}) {
        const mockFn = vi.fn()

        if (options.error) {
            mockFn.mockRejectedValue(new Error('API Error'))
        } else if (options.delay) {
            mockFn.mockImplementation(() =>
                new Promise(resolve => setTimeout(() => resolve(response), options.delay))
            )
        } else {
            mockFn.mockResolvedValue(response)
        }

        this.mocks.set(endpoint, mockFn)
        return mockFn
    }

    getMock(endpoint: string) {
        return this.mocks.get(endpoint)
    }

    clearMocks() {
        this.mocks.clear()
        vi.clearAllMocks()
    }

    expectApiCallMade(endpoint: string, times = 1) {
        const mock = this.mocks.get(endpoint)
        expect(mock).toHaveBeenCalledTimes(times)
    }

    expectApiCallWith(endpoint: string, params: any) {
        const mock = this.mocks.get(endpoint)
        expect(mock).toHaveBeenCalledWith(params)
    }
}

/**
 * State Testing Helpers for TDD
 */
export class StateTDDHelper {
    static createInitialState<T>(state: T): T {
        return { ...state }
    }

    static expectStateChange<T>(
        initialState: T,
        action: () => void,
        expectedState: Partial<T>
    ) {
        action()
        Object.keys(expectedState).forEach(key => {
            expect(initialState[key as keyof T]).toEqual(expectedState[key as keyof T])
        })
    }

    static createStateAssertion<T>(state: T) {
        return {
            toHaveProperty: (property: keyof T, value: any) => {
                expect(state[property]).toEqual(value)
            },
            toMatchObject: (expected: Partial<T>) => {
                expect(state).toMatchObject(expected)
            }
        }
    }
}

/**
 * Performance Testing Helpers for TDD
 */
export class PerformanceTDDHelper {
    static async measureRenderTime<T>(
        renderFn: () => T,
        expectedMaxTime = 100
    ): Promise<{ result: T, renderTime: number }> {
        const startTime = performance.now()
        const result = renderFn()
        const endTime = performance.now()
        const renderTime = endTime - startTime

        expect(renderTime).toBeLessThan(expectedMaxTime)

        return { result, renderTime }
    }

    static async measureAsyncOperation<T>(
        operation: () => Promise<T>,
        expectedMaxTime = 1000
    ): Promise<{ result: T, operationTime: number }> {
        const startTime = performance.now()
        const result = await operation()
        const endTime = performance.now()
        const operationTime = endTime - startTime

        expect(operationTime).toBeLessThan(expectedMaxTime)

        return { result, operationTime }
    }
}

// Export convenience functions
export const createTDDSuite = (name: string) => new TDDTestSuite(name)
export const createBehaviorSuite = () => new BehaviorTestSuite()
export const apiMock = new APIMockHelper()
export const componentTDD = ComponentTDDHelper
export const stateTDD = StateTDDHelper
export const performanceTDD = PerformanceTDDHelper 