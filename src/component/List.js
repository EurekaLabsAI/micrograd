// Filename - components/List.js

import React, { useState, useEffect, useRef } from "react";

function List() {
    const refContainer = useRef();
    const [dimensions, setDimensions] = useState({
        width: 0,
        height: 0,
    });
    useEffect(() => {
        if (refContainer.current) {
            setDimensions({
                width: refContainer.current.offsetWidth,
                height: refContainer.current.offsetHeight,
            });
        }
    }, []);
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                height: "100%",
                backgroundColor: "#fafafa",
                margin: "20px",
            }}
            ref={refContainer}
        >
            <pre>
                Container:
                <br />
                width: {dimensions.width}
                <br />
                height: {dimensions.height}
            </pre>
        </div>
    );
}

export default List;
