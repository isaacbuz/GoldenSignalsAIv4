/**
 * Neo4j Testing Utilities
 * Provides graph database testing capabilities for relationship validation
 */

import neo4j, { Driver, Session, Record } from 'neo4j-driver'
import { vi } from 'vitest'

export interface GraphTestConfig {
    uri?: string
    user?: string
    password?: string
    database?: string
}

export interface GraphNode {
    id: string
    labels: string[]
    properties: Record<string, any>
}

export interface GraphRelationship {
    type: string
    from: string
    to: string
    properties?: Record<string, any>
}

export class Neo4jTestHelper {
    private driver: Driver | null = null
    private session: Session | null = null

    constructor(private config: GraphTestConfig = {}) {
        // Use test database or mock by default
        this.config = {
            uri: config.uri || 'bolt://localhost:7687',
            user: config.user || 'neo4j',
            password: config.password || 'test',
            database: config.database || 'test',
            ...config
        }
    }

    async connect(): Promise<void> {
        try {
            this.driver = neo4j.driver(
                this.config.uri!,
                neo4j.auth.basic(this.config.user!, this.config.password!)
            )

            this.session = this.driver.session({
                database: this.config.database
            })

            // Verify connection
            await this.session.run('RETURN 1')
        } catch (error) {
            console.warn('Neo4j not available, using mock:', error)
            this.useMock()
        }
    }

    private useMock(): void {
        // Create mock implementation for testing without Neo4j
        this.driver = {
            close: vi.fn(),
            session: vi.fn(() => ({
                run: vi.fn().mockResolvedValue({ records: [] }),
                close: vi.fn()
            }))
        } as any

        this.session = {
            run: vi.fn().mockResolvedValue({ records: [] }),
            close: vi.fn()
        } as any
    }

    async disconnect(): Promise<void> {
        if (this.session) {
            await this.session.close()
        }
        if (this.driver) {
            await this.driver.close()
        }
    }

    async clearTestData(): Promise<void> {
        if (!this.session) return

        await this.session.run('MATCH (n:TestNode) DETACH DELETE n')
    }

    async createTestGraph(nodes: GraphNode[], relationships: GraphRelationship[]): Promise<void> {
        if (!this.session) return

        // Create nodes
        for (const node of nodes) {
            const labels = node.labels.map(l => `:${l}`).join('')
            const query = `
        CREATE (n${labels} {id: $id, ${Object.keys(node.properties).map(k => `${k}: $${k}`).join(', ')}})
      `
            await this.session.run(query, { id: node.id, ...node.properties })
        }

        // Create relationships
        for (const rel of relationships) {
            const query = `
        MATCH (a {id: $fromId}), (b {id: $toId})
        CREATE (a)-[r:${rel.type} ${rel.properties ? '{' + Object.keys(rel.properties).map(k => `${k}: $${k}`).join(', ') + '}' : ''}]->(b)
      `
            await this.session.run(query, {
                fromId: rel.from,
                toId: rel.to,
                ...rel.properties
            })
        }
    }

    async queryGraph(cypher: string, params: Record<string, any> = {}): Promise<Record[]> {
        if (!this.session) return []

        const result = await this.session.run(cypher, params)
        return result.records
    }

    async assertNodeExists(nodeId: string, labels?: string[]): Promise<boolean> {
        const labelFilter = labels ? labels.map(l => `:${l}`).join('') : ''
        const query = `MATCH (n${labelFilter} {id: $nodeId}) RETURN n`
        const result = await this.queryGraph(query, { nodeId })
        return result.length > 0
    }

    async assertRelationshipExists(
        fromId: string,
        toId: string,
        relationshipType: string
    ): Promise<boolean> {
        const query = `
      MATCH (a {id: $fromId})-[r:${relationshipType}]->(b {id: $toId})
      RETURN r
    `
        const result = await this.queryGraph(query, { fromId, toId })
        return result.length > 0
    }

    async getNodeDegree(nodeId: string): Promise<{ in: number, out: number, total: number }> {
        const query = `
      MATCH (n {id: $nodeId})
      OPTIONAL MATCH (n)<-[in]-()
      OPTIONAL MATCH (n)-[out]->()
      RETURN count(DISTINCT in) as inDegree, count(DISTINCT out) as outDegree
    `
        const result = await this.queryGraph(query, { nodeId })

        if (result.length === 0) {
            return { in: 0, out: 0, total: 0 }
        }

        const record = result[0]
        const inDegree = record.get('inDegree').toNumber()
        const outDegree = record.get('outDegree').toNumber()

        return {
            in: inDegree,
            out: outDegree,
            total: inDegree + outDegree
        }
    }

    async findShortestPath(fromId: string, toId: string): Promise<number> {
        const query = `
      MATCH path = shortestPath((a {id: $fromId})-[*]-(b {id: $toId}))
      RETURN length(path) as pathLength
    `
        const result = await this.queryGraph(query, { fromId, toId })

        if (result.length === 0) {
            return -1 // No path found
        }

        return result[0].get('pathLength').toNumber()
    }

    async getCentralityMetrics(nodeId: string): Promise<{
        betweenness: number
        closeness: number
        pagerank: number
    }> {
        // Note: These would require APOC procedures in a real Neo4j instance
        // For testing, we'll return mock values or simplified calculations

        const degree = await this.getNodeDegree(nodeId)

        return {
            betweenness: degree.total * 0.1, // Simplified
            closeness: degree.total > 0 ? 1 / degree.total : 0,
            pagerank: 0.15 + (0.85 * degree.in / Math.max(degree.total, 1))
        }
    }
}

// Testing utilities for graph-based components
export class GraphTestScenarios {
    constructor(private helper: Neo4jTestHelper) { }

    async createTradingSignalNetwork(): Promise<void> {
        const nodes: GraphNode[] = [
            { id: 'agent1', labels: ['Agent', 'SentimentAgent'], properties: { name: 'Sentiment Analyzer', accuracy: 0.85 } },
            { id: 'agent2', labels: ['Agent', 'TechnicalAgent'], properties: { name: 'Technical Analyzer', accuracy: 0.78 } },
            { id: 'agent3', labels: ['Agent', 'FlowAgent'], properties: { name: 'Options Flow', accuracy: 0.92 } },
            { id: 'signal1', labels: ['Signal'], properties: { type: 'BUY', confidence: 0.87, symbol: 'AAPL' } },
            { id: 'signal2', labels: ['Signal'], properties: { type: 'SELL', confidence: 0.73, symbol: 'TSLA' } },
            { id: 'market1', labels: ['Market'], properties: { symbol: 'AAPL', sector: 'Technology' } },
            { id: 'market2', labels: ['Market'], properties: { symbol: 'TSLA', sector: 'Automotive' } }
        ]

        const relationships: GraphRelationship[] = [
            { type: 'GENERATES', from: 'agent1', to: 'signal1', properties: { weight: 0.3 } },
            { type: 'GENERATES', from: 'agent2', to: 'signal1', properties: { weight: 0.4 } },
            { type: 'GENERATES', from: 'agent3', to: 'signal1', properties: { weight: 0.3 } },
            { type: 'GENERATES', from: 'agent1', to: 'signal2', properties: { weight: 0.6 } },
            { type: 'GENERATES', from: 'agent2', to: 'signal2', properties: { weight: 0.4 } },
            { type: 'TARGETS', from: 'signal1', to: 'market1' },
            { type: 'TARGETS', from: 'signal2', to: 'market2' },
            { type: 'COLLABORATES', from: 'agent1', to: 'agent2', properties: { strength: 0.7 } },
            { type: 'COLLABORATES', from: 'agent2', to: 'agent3', properties: { strength: 0.8 } }
        ]

        await this.helper.createTestGraph(nodes, relationships)
    }

    async createUserInteractionNetwork(): Promise<void> {
        const nodes: GraphNode[] = [
            { id: 'user1', labels: ['User'], properties: { email: 'test@example.com', level: 'advanced' } },
            { id: 'dashboard1', labels: ['Dashboard'], properties: { type: 'trading', layout: 'professional' } },
            { id: 'chart1', labels: ['Chart'], properties: { symbol: 'AAPL', timeframe: '1d' } },
            { id: 'ai_chat1', labels: ['AIChat'], properties: { mode: 'floating', model: 'claude' } },
            { id: 'search1', labels: ['SearchBar'], properties: { variant: 'enhanced', mode: 'hybrid' } }
        ]

        const relationships: GraphRelationship[] = [
            { type: 'VIEWS', from: 'user1', to: 'dashboard1', properties: { frequency: 'daily' } },
            { type: 'CONTAINS', from: 'dashboard1', to: 'chart1' },
            { type: 'CONTAINS', from: 'dashboard1', to: 'ai_chat1' },
            { type: 'CONTAINS', from: 'dashboard1', to: 'search1' },
            { type: 'INTERACTS_WITH', from: 'user1', to: 'chart1', properties: { actions: ['zoom', 'pan', 'indicator'] } },
            { type: 'INTERACTS_WITH', from: 'user1', to: 'ai_chat1', properties: { queries: 15, satisfaction: 0.9 } },
            { type: 'INTERACTS_WITH', from: 'user1', to: 'search1', properties: { searches: 8, success_rate: 0.85 } }
        ]

        await this.helper.createTestGraph(nodes, relationships)
    }

    async validateSignalConsensus(signalId: string): Promise<{
        agentCount: number
        averageWeight: number
        consensus: number
    }> {
        const query = `
      MATCH (a:Agent)-[r:GENERATES]->(s:Signal {id: $signalId})
      RETURN count(a) as agentCount, avg(r.weight) as avgWeight, 
             sum(r.weight * a.accuracy) / sum(r.weight) as consensus
    `
        const result = await this.helper.queryGraph(query, { signalId })

        if (result.length === 0) {
            return { agentCount: 0, averageWeight: 0, consensus: 0 }
        }

        const record = result[0]
        return {
            agentCount: record.get('agentCount').toNumber(),
            averageWeight: record.get('avgWeight'),
            consensus: record.get('consensus')
        }
    }

    async validateComponentRelationships(componentType: string): Promise<{
        totalComponents: number
        connectedComponents: number
        isolatedComponents: number
        averageConnections: number
    }> {
        const query = `
      MATCH (c:${componentType})
      OPTIONAL MATCH (c)-[r]-()
      RETURN c.id as componentId, count(r) as connections
    `
        const result = await this.helper.queryGraph(query)

        const components = result.map(r => ({
            id: r.get('componentId'),
            connections: r.get('connections').toNumber()
        }))

        const totalComponents = components.length
        const connectedComponents = components.filter(c => c.connections > 0).length
        const isolatedComponents = totalComponents - connectedComponents
        const averageConnections = components.reduce((sum, c) => sum + c.connections, 0) / totalComponents

        return {
            totalComponents,
            connectedComponents,
            isolatedComponents,
            averageConnections
        }
    }
}

// Export singleton instance for easy testing
export const graphTestHelper = new Neo4jTestHelper()
export const graphScenarios = new GraphTestScenarios(graphTestHelper)

// Vitest setup helpers
export async function setupGraphTests() {
    await graphTestHelper.connect()
    await graphTestHelper.clearTestData()
}

export async function teardownGraphTests() {
    await graphTestHelper.clearTestData()
    await graphTestHelper.disconnect()
} 