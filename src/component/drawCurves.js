import styles from './classes.module.css';
drawCurves = () => {
    // Here we remove all existing links in the DOM before the new render of the modified tree
    document
        .querySelectorAll(`div[class*="${styles.svg_container}"]`)
        .forEach((e) => e.remove())
    // the pipelineContainer contains a root element to display the entire tree
    const pipelineContainer = document.getElementById('tree_wrapper')
    const { nodeReferencies } = this.state
    this.state.columns.forEach((column, id) => {
        column.forEach((cell, index) => {
            // Exclude execution for the last layer in the structure
            if (id !== this.state.columns.length - 1) {
                const cardElement = ReactDOM.findDOMNode(
                    nodeReferencies[id][index].element
                )
                const childrenElements = []
                this.state.columns[id + 1].map((child, c) => {
                    if (child && child.parent_node === cell.node) {
                        childrenElements.push(nodeReferencies[id + 1][c].element)
                    }
                })
                // Here we will store the coordinates of the middle of the left side of the child
                const coords = []
                if (childrenElements.length) {
                    childrenElements.forEach((ce) => {
                        if (ce) {
                            const domNode = ReactDOM.findDOMNode(ce)
                            if (domNode)
                                coords.push({
                                    x: domNode.offsetLeft,
                                    y: domNode.offsetTop + domNode.offsetHeight / 2
                                })
                        }
                    })
                    if (cardElement) {
                        const x = cardElement.offsetWidth + cardElement.offsetLeft
                        // Get the midpoint of the parent's right edge
                        const center =
                            cardElement.offsetTop + cardElement.offsetHeight / 2
                        if (x !== 0 && !cardElement.classList.contains('add_card')) {
                            // Draw a curve
                            const svgContainer = document.createElement('div')
                            svgContainer.setAttribute('class', styles.svg_container)
                            const circle = document.createElementNS(
                                'http://www.w3.org/2000/svg',
                                'svg'
                            )
                            const checkPoints = coords
                                .map((coord) => coord.y)
                                .concat(center)
                            const height =
                                _.max(checkPoints) - _.min(checkPoints) >= 12
                                    ? _.max(checkPoints) - _.min(checkPoints)
                                    : 12
                            circle.setAttribute('width', 160)
                            circle.setAttribute('height', height)
                            circle.setAttribute('viewBox', `0 0 160 ${height}`)
                            circle.setAttribute('version', '1.1')
                            circle.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
                            let sourcePositionY = null
                            if (Math.abs(_.min(checkPoints) - center) === height) {
                                sourcePositionY = height - 6
                            } else if (Math.abs(_.min(checkPoints) - center) < 6) {
                                sourcePositionY = 6
                            } else {
                                sourcePositionY = Math.abs(_.min(checkPoints) - center)
                            }
                            const lines = coords.map((coord) => {
                                const besie = `<path d="M 6, ${sourcePositionY} C 80, ${sourcePositionY}, 80, ${Math.abs(
                                    coord.y - _.min(checkPoints)
                                )}, 160, ${Math.abs(
                                    coord.y - _.min(checkPoints)
                                )}" stroke="#C4C4C4" fill="none"/>`
                                return besie
                            })
                            circle.innerHTML = `
                      ${lines.join('\n')}
                      ${lines.length
                                    ? `<circle cx="6" cy=${sourcePositionY} r="4" fill="#C4C4C4"/>`
                                    : null
                                }
                    `
                            svgContainer.style.top = _.min(checkPoints) + 'px'
                            svgContainer.style.left = x + 'px'
                            svgContainer.appendChild(circle)
                            pipelineContainer.appendChild(svgContainer)
                        }
                    }
                }
            }
        })
    })
}