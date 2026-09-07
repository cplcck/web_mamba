export function connectPaneResize(workspace: HTMLElement, handle: HTMLElement, compact: MediaQueryList): () => void {
  let preferred: number | null = null
  let width = 0
  let range = { min: 0, max: 0, step: 0 }
  let drag: Readonly<{ id: number; x: number; width: number }> | null = null
  const root = workspace.ownerDocument.documentElement

  const writeWidth = (value: number, remember: boolean): void => {
    width = Math.round(Math.max(range.min, Math.min(range.max, value)))
    if (remember) preferred = width
    workspace.style.setProperty('--inspector-width', `${width}px`)
    handle.setAttribute('aria-valuemin', String(range.min))
    handle.setAttribute('aria-valuemax', String(range.max))
    handle.setAttribute('aria-valuenow', String(width))
    handle.setAttribute('aria-valuetext', `선택 정보 ${width}px`)
  }
  const finish = (): void => {
    const active = drag
    drag = null
    root.classList.remove('pane-resizing')
    if (active && handle.hasPointerCapture(active.id)) handle.releasePointerCapture(active.id)
  }
  const reflow = (): void => {
    handle.hidden = compact.matches
    handle.tabIndex = compact.matches ? -1 : 0
    handle.setAttribute('aria-disabled', String(compact.matches))
    if (compact.matches) { finish(); return }
    const style = getComputedStyle(workspace)
    const minimum = parseFloat(style.getPropertyValue('--inspector-width-min'))
    const explorerMinimum = parseFloat(style.getPropertyValue('--explorer-width-min'))
    const initial = parseFloat(style.getPropertyValue('--inspector-width-wide'))
    const step = parseFloat(style.getPropertyValue('--space-4'))
    // The stylesheet may finish after data; ResizeObserver also sees that layout.
    if (![minimum, explorerMinimum, initial, step].every(Number.isFinite)) return
    const available = workspace.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    const maximum = Math.max(minimum, Math.floor(available - handle.getBoundingClientRect().width - explorerMinimum))
    if (drag && maximum !== range.max) finish()
    range = { min: minimum, max: maximum, step }
    preferred ??= initial
    writeWidth(preferred, false)
  }

  handle.addEventListener('pointerdown', event => {
    if (compact.matches || event.button !== 0 || !event.isPrimary || drag) return
    event.preventDefault()
    drag = { id: event.pointerId, x: event.clientX, width }
    handle.setPointerCapture(event.pointerId)
    handle.focus({ preventScroll: true })
    root.classList.add('pane-resizing')
  })
  handle.addEventListener('pointermove', event => {
    if (drag?.id === event.pointerId) writeWidth(drag.width + drag.x - event.clientX, true)
  })
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) {
    handle.addEventListener(type, event => { if (drag?.id === event.pointerId) finish() })
  }
  handle.addEventListener('keydown', event => {
    if (compact.matches) return
    let next: number
    switch (event.key) {
      case 'ArrowLeft': next = width + range.step; break
      case 'ArrowRight': next = width - range.step; break
      case 'Home': next = range.min; break
      case 'End': next = range.max; break
      default: return
    }
    event.preventDefault()
    writeWidth(next, true)
  })
  workspace.ownerDocument.defaultView?.addEventListener('blur', finish)
  new ResizeObserver(reflow).observe(workspace)
  reflow()
  return reflow
}
