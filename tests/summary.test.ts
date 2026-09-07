import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { buildInspectorModel, buildInspectorSummary } from '../src/inspector'
import { validateDocument } from '../src/schema'

const documents = ['prefill', 'decode'].map(scenario => validateDocument(JSON.parse(readFileSync(`public/data/${scenario}.json`, 'utf8'))))

describe('concise selected tensor summaries', () => {
  it.each(documents)('preserves actual counts and bounded meaningful tensors for $scenario.name', document => {
    // Given: every actual model/block/stage/operator boundary, not a summary fixture.
    for (const entity of document.entities) {
      const model = buildInspectorModel(document, entity.id)
      // When
      const summary = buildInspectorSummary(model)
      // Then: total counts are honest; state never masquerades as the main activation.
      for (const [index, section] of summary.sections.entries()) {
        const full = model.sections[index]
        expect(section.count).toBe(full?.count)
        expect(section.preview.length).toBeLessThanOrEqual(2)
        expect(section.preview.every(tensor => full?.tensors.some(actual => actual.id === tensor.id))).toBe(true)
        expect(section.preview.some(tensor => tensor.role === 'state')).toBe(false)
        if (full?.tensors.some(tensor => tensor.role === 'activation')) expect(section.preview.every(tensor => tensor.role === 'activation')).toBe(true)
        expect(section.stateCount).toBe(full?.tensors.filter(tensor => tensor.role === 'state').length)
        expect(section.uniqueGgufPayloadBytes).toBe(full?.uniqueGgufPayloadBytes)
      }
      expect(summary.state.preview.length).toBeLessThanOrEqual(2)
      expect(summary.state.preview.every(tensor => tensor.role === 'state')).toBe(true)
    }
  })

  it('shows the selected block activation and model-wide root rather than first-block data', () => {
    // Given
    const document = documents[0]
    if (!document) throw new Error('missing capture')
    // When
    const block = buildInspectorSummary(buildInspectorModel(document, 'block.23'))
    const root = buildInspectorSummary(buildInspectorModel(document, 'mamba-130m'))
    // Then
    expect(block.sections[0]?.preview.map(tensor => tensor.name)).toEqual(['l_out-22'])
    expect(block.sections[1]?.preview.map(tensor => tensor.name)).toEqual(['l_out-23'])
    expect(root.sections[0]?.preview.map(tensor => tensor.name)).toEqual(['inp_tokens'])
    expect(root.state.preview).toEqual([])
  })
})
