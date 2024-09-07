const createColumnsData = (tree) => {
    const columns = []
    const nodeReferencies = []
    const newTree = {}
    Object.assign(newTree, tree)
    const treeWalker = (treeNode, column = 0, parent = null) => {
        if (!columns[column]) {
            columns.push([])
            nodeReferencies.push([])
        }
        if (!columns[column + 1]) {
            columns.push([])
            nodeReferencies.push([])
        }
        treeNode.parent_node = parent ? parent.node : null
        treeNode.node = `${column}${columns[column].length}`
        columns[column].push(treeNode)
        nodeReferencies[column].push({
            parent_node: parent ? parent.node : null,
            node: treeNode
        })
        if (treeNode.children) {
            treeNode.children.forEach((child, id) => {
                if (child === null) {
                    treeNode.children[id] = {
                        children: [],
                        parent_node: treeNode,
                        node: `${column}${columns[column].length}`
                    }
                }
                treeWalker(treeNode.children[id], column + 1, treeNode)
            })
        }
    }
    treeWalker(newTree)
    return {
        columns,
        nodeReferencies
    }
}