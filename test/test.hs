
import AI.SparseCoder
import Data.Packed.Matrix
import Data.Packed.Vector

f x = 1.0 / ( 1.0 + exp(-x))
f' x = exp(x) / ( (1.0 + exp x) ^ 2 )

createVector:: Int -> Vector Double
createVector x = fromList $ map (\y -> fromIntegral (x `mod` 10) / 10.0) [1..10]

main = do
  let sc = create 10 2 f f'
  let xs = map createVector [1..1000]
  let (sc', g) = trainN sc [] xs 500
  --putStrLn $ show $ head g 
  putLst $ reverse g
  --putStrLn $ show $ head xs 

putLst [] = return ()
putLst (x:xs) = do
  putStrLn $ show  x
  putLst xs

main2 = do
  let sc = create 10 7 f f'
  let xs = map createVector[1..100]
  sc' <- go sc xs 500
  putStrLn $ show $ squaredError sc' xs

go :: SparseCoder -> [Vector Double] -> Int -> IO SparseCoder
go sc _ 0 = return sc

go sc xs i = do
  let sc' = train sc xs
  let se = squaredError sc' xs
  --putStrLn $ show $ se
  se `seq` go sc' xs (i-1)
