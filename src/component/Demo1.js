import { useRef, useEffect } from 'react';
const Canvas = props => {
    const ref = useRef();
    return <canvas ref={ref} {...props} />
}
export default Canvas;