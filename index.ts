interface BoosterNode {
  nodeid: number,
  depth: number,
  split: string,
  split_condition: number,
  yes: number,
  no: number,
  missing: number,
  children: Array<BoosterNode | BoosterLeaf>
}

interface BoosterLeaf {
  nodeid: number,
  leaf: number
}

type Booster = BoosterNode; // The root of the tree
type XGBoostModel = Array<Booster>;
type FeatureIndex = Record<string, number>;
type ReverseFeatureIndex = Record<string, string>;

async function loadJson(file: string) {
  const fs = await import("fs");
  const buffer = await fs.promises.readFile(file);
  return JSON.parse(buffer.toString());
}

function sigmoid(x: number) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

function isLeaf(node: BoosterNode | BoosterLeaf): node is BoosterLeaf {
  return (node as BoosterLeaf).leaf !== undefined;
}

export class Scorer {
  model?: XGBoostModel;
  reverseFeatureIndex?: ReverseFeatureIndex;

  static async create(model: string | object, featureIndex?: string | object) {
    const scorer = new Scorer
    scorer.model = typeof model === "string" ? await loadJson(model) : model;
    if (featureIndex) {
      const loadedFeatureIndex: FeatureIndex =
        typeof featureIndex === "string" ? await loadJson(featureIndex) : featureIndex;
      scorer.reverseFeatureIndex =
        Object.keys(loadedFeatureIndex)
          .reduce((acc: Record<string, string>, fName: string) => {
            const fIdx: number = loadedFeatureIndex[fName];
            acc[`${fIdx}`] = fName;
            return acc;
          }, {});
    }
    return scorer;
  }

  scoreSingleInstance(features: Record<string, number>) {
    if (!this.model) {
      throw new Error(`Scorer not initialized, create a scorer using Scorer.create() only`)
    }
    const totalScore: number =
      this.model
        .map((booster: Booster) => {
          let currNode: BoosterNode | BoosterLeaf = booster;
          while (!isLeaf(currNode)) {
            const splitFeature = currNode.split;
            let nextNodeId: number;
            if (features[splitFeature] !== undefined) {
              const conditionResult = features[splitFeature] < currNode.split_condition;
              nextNodeId = conditionResult ? currNode.yes : currNode.no;
            } else {
              // Missing feature
              nextNodeId = currNode.missing;
            }
            const nextNode: BoosterNode | BoosterLeaf | undefined =
              currNode.children.find(child => child.nodeid === nextNodeId);
            if (nextNode === undefined) {
              throw new Error(`Invalid model JSON, missing node ID: ${nextNodeId}`)
            }
            currNode = nextNode;
          }
          return currNode.leaf;
        })
        .reduce((score, boosterScore) => score + boosterScore, 0.0)
    return sigmoid(totalScore);
  }

  async score(input: string | object | Array<object>): Promise<Array<number> | number> {
    if (typeof input !== "string" && typeof input !== "object") {
      throw new Error(`Invalid input to score method: ${input}, expected string or object, was ${typeof input}`)
    }

    // Scoring a single instance or array of instances
    if (typeof input === "object") {
      if (Array.isArray(input)) {
        return (input as Array<object>).map(en => this.scoreSingleInstance(en as Record<string, number>));
      } else {
        return this.scoreSingleInstance(input as Record<string, number>);
      }
    }

    if (!this.reverseFeatureIndex) {
      throw new Error(`Cannot score LibSVM input without a feature index, please specify one while creating a scorer.`)
    }

    // Scoring a LibSVM data file
    const fs = await import("fs");
    const readline = await import("readline");
    const inputStream = fs.createReadStream(input);
    const rl = readline.createInterface({
      input: inputStream,
      crlfDelay: Infinity
    })

    let scores = [];
    for await (const line of rl) {
      const features: Record<string, number> =
        line
          .split(" ")
          .slice(1)
          .map(p => p.split(":"))
          .map(([featureId, value]) => [(this.reverseFeatureIndex as ReverseFeatureIndex)[featureId], value])
          .reduce((featureMap: Record<string, number>, entry: Array<string>) => {
            const [ featureName, featureValue ] = entry;
            featureMap[featureName] = parseFloat(featureValue);
            return featureMap;
          }, {});
      const score = this.scoreSingleInstance(features);
      scores.push(score);
    }
    return scores;
  }
}
