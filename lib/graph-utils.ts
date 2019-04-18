export class GraphUtil {

  private vertices: number;
  private color: boolean[];
  private tensors: Map<number, { from: Set<number>, to: Set<number> }>;
  private tensorMapping: number[][];
  private next: number[][];
  private prev: number[][];
  private inTensorsOfInputNode: Map<number, Set<number>>;
  private outTensorsOfOutputNode: Map<number, Set<number>>;

  /**
   * Create a directed graph that supports partitioning by two colors.
   * @param {number} vertices     Number of vertices in the graph.
   */
  constructor(vertices: number) {

    // Partitioning a neural network graph requires two graphs:
    // a "node graph" and a "tensor graph"

    // The following properties describe a "node graph", in which nodes are
    // identified by a unique id of type `number`
    //   private vertices: number;            // number of ndoes
    //   private color: boolean[];            // false - white, true - black
    //   private next: number[][];            // adjacency list
    //   private prev: number[][];            // reversed adjacency list

    // The following properties describe a "tensor graph", in which tensors are
    // identified by a unique id of type `number`
    //   // head and tail nodes of each tensor
    //   private tensors: Map<number, { from: Set<number>, to: Set<number> }>;
    //   // tensors from node i to j
    //   private tensorMapping: number[][];
    //   private inTensorsOfInputNode: Map<number, Set<number>>;
    //   private outTensorsOfOutputNode: Map<number, Set<number>>;

    this.vertices = vertices;
    this.color = new Array(vertices).fill(false);
    this.next = [];
    this.prev = [];

    this.tensors = new Map();
    this.tensorMapping = [];
    this.inTensorsOfInputNode = new Map();
    this.outTensorsOfOutputNode = new Map();

    for (let i = 0; i < vertices; i++) {
      this.next[i] = [];
      this.prev[i] = [];
      this.tensorMapping[i] = [];
    }
  }

  /**
   * Add an edge between node `i` and `j` with only one `tensor` attached on it
   *
   * This is a helper of the function `addNode`. It can be used for constructing
   * a directed graph that supports topological sorting with cycle detection
   *
   * @param {number} i            Head node
   * @param {number} j            Tail node
   * @param {number} [tensor=-1]  Tensor attached on edge i -> j
   *
   */
  addEdge(i: number, j: number, tensor?: number) {
    // at most one tensor attached to edge i->j
    this.next[i].push(j);
    this.prev[j].push(i);
    this.tensorMapping[i][j] = typeof tensor !== 'undefined' ? tensor : -1;
  }

  /**
   * Add a neural network node with input and output tensors
   *
   * @param {number}   nodeId     Node within the range [0, vertices-1]
   * @param {number[]} inTensors  List of input tensors of the node
   * @param {number[]} outTensors List of output tensors of the node
   *
   */
  addNode(nodeId: number, inTensors: number[], outTensors: number[]) {

    for (const i of inTensors) {
      if (!this.tensors.has(i)) {
        this.tensors.set(i, {
          from: new Set(),
          to: new Set()
        });
      }
      for (const inNodeId of this.tensors.get(i)!.from) {
        this.addEdge(inNodeId, nodeId, i);
      }
      this.tensors.get(i)!.to.add(nodeId);
    }

    for (const i of outTensors) {
      if (!this.tensors.has(i)) {
        this.tensors.set(i, {
          from: new Set(),
          to: new Set()
        });
      }
      for (const outNodeId of this.tensors.get(i)!.to) {
        this.addEdge(nodeId, outNodeId, i);
      }
      this.tensors.get(i)!.from.add(nodeId);
    }
  }

  /**
   * Mark a node black
   *
   * @param  {number} i           Node to be marked black
   */
  setBlack(i: number) {
    this.color[i] = true;
  }

  /**
   * Identify the input and output tensors of the whole graph
   *
   * @param {number[]} inTensors  List of input tensors of the graph
   * @param {number[]} outTensors List of output tensors of the graph
   */
  identifyInputOutputTensors(inTensors: number[], outTensors: number[]) {

    for (const t of inTensors) {
      if (!this.tensors.has(t)) {
        return;
      }
      for (const n of this.tensors.get(t)!.to) {
        if (!this.inTensorsOfInputNode.has(n)) {
          this.inTensorsOfInputNode.set(n, new Set());
        }
        this.inTensorsOfInputNode.get(n)!.add(t);
      }
    }

    for (const t of outTensors) {
      if (!this.tensors.has(t)) {
        return;
      }
      for (const n of this.tensors.get(t)!.from) {
        if (!this.outTensorsOfOutputNode.has(n)) {
          this.outTensorsOfOutputNode.set(n, new Set());
        }
        this.outTensorsOfOutputNode.get(n)!.add(t);
      }
    }

  }

  /**
   * Topological sorting
   *
   * @returns {number[]} List of nodes in topological order
   */
  topologicalSort(): number[] {
    const indegree = new Array(this.vertices).fill(0);
    const result: number[] = [];
    const q: number[] = [];
    for (let i = 0; i < this.vertices; i++) {
      indegree[i] = this.prev[i].length;
      if (!indegree[i]) {
        q.push(i); // push node i with indegree zero
      }
    }

    let cnt = 0;
    while (q.length) {
      const u = q.shift()!;
      result.push(u);
      cnt++;
      for (const v of this.next[u]) {
        if (!--indegree[v]) {
          q.push(v);
        }
      }
    }

    if (cnt !== this.vertices) {
      throw new Error('Not a DAG');
    }
    return result;
  }

  /**
   * Partition the grash by two colors. It returns a unique solution based on
   * the given dichromatic graph.
   *
   * @returns {Set<number>[]} A list of "partition sets". In each set, all nodes
   *                          have the same color and no order. Each set is
   *                          dependent on the previous set. Specifically, if
   *                          all nodes are of the same color, there'll be only
   *                          one "partition set" with all nodes in it.
   *
   * @example
   *              xxxxx                      xxxxx              ooooo
   *              x 0 x                      x   x : black      o   o : white
   *              xxxxx                      xxxxx              ooooo
   *      __________|__________
   *     |          |          |
   *   ooooo      ooooo      xxxxx
   *   o 1 o      o 2 o      x 3 x
   *   ooooo      ooooo      xxxxx           results:
   *     |          |          |             [{0, 3}, {1, 2, 4, 6}, {5, 7}]
   *   ooooo      xxxxx      ooooo
   *   o 4 o      x 5 x      o 6 o
   *   ooooo      xxxxx      ooooo
   *     |_________ | _________|
   *               \|/
   *              xxxxx
   *              x 7 x
   *              xxxxx
   *
   */
  biTopologicalSort(): Set<number>[] {
    const order = new Array(this.vertices).fill(0);
    for (const u of this.topologicalSort()) {
      for (const v of this.prev[u]) {
        if (this.color[u] === this.color[v]) {
          order[u] = Math.max(order[u], order[v]);
        } else {
          order[u] = Math.max(order[u], order[v] + 1);
        }
      }
    }

    const result = [];
    for (const [nodeId, ord] of order.entries()) {
      if (typeof result[ord] === 'undefined') {
        result[ord] = new Set();
      }
      result[ord].add(nodeId);
    }
    return result;
  }

  /**
   * Extend the "partition sets" returned by `biTopologicalSort`.
   *
   * Resolve the input and output tensors of each "partition set". These tensors
   * lie on the cross edges, which are shared by two connected partitions.
   *
   * @param  {boolean} [eager=false]  Partition in eager mode
   *                                  i.e. each node belongs to one partition set
   *
   * @typedef  {Object} PartitionResults
   * @property {Set<number>[]}  nodes A partition set
   * @property {number[]}   inTensors Input tensors of the partition set
   * @property {number[]}  outTensors Output tensors of the partition set
   *
   * @returns {PartitionResults} Resolve the input and output tensors of each sets
   *                          returned by `biTopologicalSort`
   */
  partition(eager = false) {

    function union<T>(a: Set<T>, b: Set<T>): Set<T> {
      return new Set([...a, ...b]);
    }

    function sortSet(set: Set<number>) {
      return Array.from(set).sort((a, b) => a - b);
    }

    const result = [];
    // crossTensor - tensor lies on the cross edge
    const crossTensorsTo = new Map();
    for (let i = 0; i < this.vertices; i++) {
      crossTensorsTo.set(i, new Set());
    }

    let partitions = [];
    if (eager) {
      for (const i of this.topologicalSort())
        partitions.push(new Set([i]));
    } else {
      partitions = this.biTopologicalSort();
    }

    for (const partition of partitions) {
      let inTensors = new Set();
      let outTensors = new Set();
      for (const u of partition) {
        for (const v of this.next[u]) {
          if (!partition.has(v)) {
            const tensorUV = this.tensorMapping[u][v];
            crossTensorsTo.get(v).add(tensorUV);
            outTensors.add(tensorUV);
          }
        }

        if (this.outTensorsOfOutputNode.has(u)) {
          outTensors = union(outTensors, this.outTensorsOfOutputNode.get(u)!);
        }
      }
      for (const u of partition) {
        inTensors = union(inTensors, crossTensorsTo.get(u)!);

        if (this.inTensorsOfInputNode.has(u)) {
          inTensors = union(inTensors, this.inTensorsOfInputNode.get(u)!);
        }
      }

      result.push({
        nodeIds: sortSet(partition),
        inputIds: sortSet(inTensors),
        outputIds: sortSet(outTensors),
      });
    }
    return result;
  }
}
