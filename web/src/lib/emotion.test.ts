import { describe, expect, it } from "vitest";
import { lexicalPrediction } from "./emotion";

describe("lexicalPrediction", () => {
  it("returns a normalized six-way emotion distribution", () => {
    const prediction = lexicalPrediction("I am furious and angry about this unfair decision");
    expect(prediction.emotion).toBe("angry");
    expect(Object.values(prediction.scores).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    expect(prediction.backend).toBe("lexical-fallback");
  });

  it("uses a uniform distribution for unknown language", () => {
    const prediction = lexicalPrediction("something entirely ambiguous");
    expect(prediction.confidence).toBeCloseTo(1 / 6);
  });
});
