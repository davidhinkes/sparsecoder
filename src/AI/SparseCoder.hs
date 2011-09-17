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
  where randomMatrix x y = (x><y) $ randomRs (-0.01,0.01) (mkStdGen 0)

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
augment x = buildVector (dim x + 1) (augment' x) where
  augment' x i = if (dim x) == i then 1.0 else x @> i

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
trainOnce :: SparseCoder -> Vector Double -> Vector Double -> SparseCoder
trainOnce sc p x =
  let alpha = 0.01
      f = activationFunction sc
      f' = activationFunction' sc
      w_o = outputLayer sc
      w_h = hiddenLayer sc
      (z_h, z_o) = evalWithLayers sc x
      a_i = x
      a_h = mapVector f z_h
      a_o = mapVector f z_o
      ie =  a_o `sub` x -- optimization, could have used instanceError
      d_o = ie `mul` (mapVector f' z_o)
      d_h = (deaugment $ (trans w_o `mXv` d_o)) `mul` (mapVector f' z_h)
      deltaW_o = asColumn d_o `mXm` (asRow $ augment a_h)
      deltaW_h =  (asColumn d_h) `mXm` (asRow $ augment a_i)
  in SparseCoder (w_h `sub` (scale alpha deltaW_h)) (w_o `sub` (scale alpha deltaW_o)) f f'

-- Train network based on many examples.
train :: SparseCoder -> Vector Double -> [Vector Double] -> SparseCoder
train sc p (x:xs) = let sc' = trainOnce sc p x in sc' `seq` p `seq`  train sc' p xs
train sc _ [] = sc

getP :: SparseCoder -> [Vector Double] -> Vector Double
getP sc xs =
  let n = rows $ hiddenLayer sc
      zerroVector = fromList $ replicate n 0.0 in
  scale (1.0/(fromIntegral n)) $ getP' sc zerroVector xs
getP' :: SparseCoder -> Vector Double -> [Vector Double] -> Vector Double
getP' _ p [] = p
getP' sc p (x:xs) =
  let (z_h, _) = evalWithLayers sc x
      a_h = mapVector (activationFunction sc) z_h in
    getP' sc (a_h `add` p) xs

-- Train many times.
trainN :: SparseCoder -> [Double] -> [Vector Double] -> Int -> (SparseCoder, [Double])
trainN sc e _  0 = (sc, e)
trainN sc e xs n =
  let p = getP sc xs
      sc' = train sc p xs
      se = squaredError sc' xs
  in se `seq` trainN sc' (se:e) xs (n-1)
