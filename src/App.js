import Graphviz from './component/Graphviz';
import Demo from './component/Demo';
import Optimizer from './component/Optimizer';
import React from 'react';
import { ChakraProvider, Grid, GridItem, Text } from '@chakra-ui/react';

function App() {
  return (
    <ChakraProvider>

      <Grid className="container12"
        templateAreas={`"gv gv"
                        "gv gv"
                        "gr op"
                        "gr op"`}
        gridTemplateRows={'1fr 1fr'}
        gridTemplateColumns={'1fr 1fr'}
        h='calc(100vh)'
        gap='1'
        color='blackAlpha.700'
        fontWeight='bold'
        bg='gray.100'
        p='4'
        borderRadius='md'
      >
        <GridItem
          pl='2'
          bg='blue.400'
          area={'gv'}
          borderRadius='md'
          color='white'
          fontSize='1.2rem'
          display='flex'
          m="1"
          alignItems='center'
        >
          <Graphviz />
        </GridItem>
        <GridItem
          pl='2'
          bg='pink.400'
          area={'gr'}
          borderRadius='md'
          p='2'
          m="1"
          color='white'
          fontSize='1.2rem'

        >
          <Demo />
        </GridItem>
        <GridItem
          pl='2'
          bg='green.400'
          area={'op'}
          borderRadius='md'
          p='2'
          m="1"
          color='white'
          fontSize='1.2rem'
        >
          <Optimizer />
        </GridItem>
      </Grid>
    </ChakraProvider>
  );
}

export default App;
