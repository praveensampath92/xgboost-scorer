import * as fs from "fs";
import * as readline from "readline";

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

function loadJsonSync(file: string) {
  const buffer = fs.readFileSync(file);
  return JSON.parse(buffer.toString());
}

function sigmoid(x: number) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

function isLeaf(node: BoosterNode | BoosterLeaf): node is BoosterLeaf {
  return (node as BoosterLeaf).leaf !== undefined;
}

export class Scorer {
  model: XGBoostModel;
  reverseFeatureIndex: ReverseFeatureIndex;

  constructor(modelFile: string, featureIndexFile: string) {
    this.model = loadJsonSync(modelFile);
    const featureIndex: FeatureIndex = loadJsonSync(featureIndexFile);
    this.reverseFeatureIndex =
      Object.keys(featureIndex)
        .reduce((acc: Record<string, string>, fName: string) => {
          const fIdx: number = featureIndex[fName];
          acc[`${fIdx}`] = fName;
          return acc;
        }, {});
  }

  scoreSingleInstance(features: Record<string, number>) {
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

  async score(dataFile: string) {
    const inputStream = fs.createReadStream(dataFile);
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
          .map(([featureId, value]) => [this.reverseFeatureIndex[featureId], value])
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
