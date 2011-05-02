module AI.SparseCoder (
  SparseCoder(..),
  create,
  eval,
  train,
  trainN,
  squaredError,
) where

import Data.Packed.Matrix
import Data.Packed.Vector
import Numeric.Container
import System.Random

type ActivationFunction = (Double -> Double)

data SparseCoder = SparseCoder {
  hiddenLayer :: (Matrix Double),
  outputLayer :: (Matrix Double),
  activationFunction :: ActivationFunction,
  activationFunction' :: ActivationFunction
}

create :: Int -> Int -> ActivationFunction -> ActivationFunction -> SparseCoder
create input hidden f f' = SparseCoder (randomMatrix hidden (input+1)) (randomMatrix input (hidden+1)) f f'
  where randomMatrix x y = (x><y) $ randomRs (-0.02,0.02) (mkStdGen 0)

-- Evaluate ANN layers pre-activation function given input vector.
evalWithLayers :: SparseCoder -> Vector Double -> (Vector Double, Vector Double)
evalWithLayers sc input =
  let z_h = (hiddenLayer sc) `mXv` (augment input) 
      a_h = mapVector (activationFunction sc) z_h
      z_o = (outputLayer sc) `mXv` (augment a_h)
  in (z_h, z_o)

eval :: SparseCoder -> Vector Double -> Vector Double
eval sc input = let (_, z_o) = evalWithLayers sc input in
  mapVector (activationFunction sc) z_o

-- Increase dimention of vector by one, appending a 1.0 value.
augment :: Vector Double -> Vector Double
augment x = fromList $ (toList x) ++ [1.0]

deaugment :: Vector Double -> Vector Double
deaugment x = subVector 0 (dim x - 1) x

-- Error, in vector form, from a single training instance.
instanceError :: SparseCoder -> Vector Double -> Vector Double
instanceError sc x = sub x (eval sc x)

-- J: sum-squared error from all training instances.
squaredError :: SparseCoder -> [Vector Double] -> Double
squaredError sc xs = squaredError' sc xs 0.0 where
  squaredError' _ [] e = e
  squaredError' sc (x:xs) e =
    let ie = instanceError sc x
    in squaredError' sc xs (e + dot ie ie)

-- Train network based on single example.
trainOnce :: SparseCoder -> Vector Double -> SparseCoder
trainOnce sc x =
  let alpha = 0.01
      f = activationFunction sc
      f' = activationFunction' sc
      w_o = outputLayer sc
      w_h = hiddenLayer sc
      (z_h, z_o) = evalWithLayers sc x
      a_i = augment x
      a_h = mapVector f z_h
      a_o = mapVector f z_o
      ie =  x `sub` a_o -- optimization, could have used instanceError
      d_o = ie `mul` (mapVector f' z_o)
      d_h = (trans w_o `mXv` d_o) `mul` (augment $ mapVector f' z_h)
      --deltaW_o = (scale alpha $ diag d_o) `mXm` w_o
      deltaW_o = asColumn d_o `mXm` (asRow $ augment a_h)
      --deltaW_h = (scale alpha $ diag $ deaugment d_h) `mXm` w_h
      deltaW_h =  asColumn (deaugment d_h) `mXm` (asRow a_i)
  in SparseCoder (w_h `add` deltaW_h) (w_o `add` deltaW_o) f f'

-- Train network based on many examples.
train :: SparseCoder -> [Vector Double] -> SparseCoder
train sc (x:xs) = let sc' = trainOnce sc x in train sc' xs
train sc [] = sc

-- Train many times.
trainN :: SparseCoder -> [Double] -> [Vector Double] -> Int -> (SparseCoder, [Double])
trainN sc e _  0 = (sc, e)
trainN sc e xs n =
  let sc' = train sc xs
      se = squaredError sc' xs
  in se `seq` trainN sc' (se:e) xs (n-1)
