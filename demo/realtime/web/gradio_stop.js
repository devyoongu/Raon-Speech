async (sessionId) => {
  const state = window.__raon_fd_demo || {};
  if (state.ws && state.ws.readyState <= 1) {
    try {
      const closePayload = new TextEncoder().encode("finish");
      const frame = new Uint8Array(1 + closePayload.length);
      frame[0] = 0x06;
      frame.set(closePayload, 1);
      state.ws.send(frame);
    } catch (_) {}
    try { state.ws.close(1000, "finish"); } catch (_) {}
  }
  if (state.processor) {
    try { state.processor.disconnect(); } catch (_) {}
  }
  if (state.micSource) {
    try { state.micSource.disconnect(); } catch (_) {}
  }
  if (state.stream) {
    try { state.stream.getTracks().forEach((track) => track.stop()); } catch (_) {}
  }
  if (state.captureContext) {
    try { await state.captureContext.close(); } catch (_) {}
  }
  if (state.sources) {
    for (const src of state.sources) {
      try { src.stop(0); } catch (_) {}
      try { src.disconnect(); } catch (_) {}
    }
    try { state.sources.clear(); } catch (_) {}
  }
  if (state.audioContext) {
    try { await state.audioContext.close(); } catch (_) {}
  }
  window.__raon_fd_demo = {};
  return [sessionId];
}
