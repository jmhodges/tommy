#!/usr/bin/env ruby

require 'set'

module Tommy
class BayesData < Hash
  attr_reader :name, :training, :pool
  attr_accessor :token_count, :train_count
  def initialize(name='', pool=nil)
    @name = name
    @training = []
    @pool = pool
    @token_count = 0
    @train_count = 0
  end

  def trained_on?(item)
    training.include? item
  end

  def inspect
    "#<BayesDict: #{name}, #{token_count}"
  end
end

class Bayes

  attr_reader :dirty, :pools, :data_class, :combiner, :tokenizer

  # opts can include :tokenizer, :combiner or :data_class
  def initialize(opts = {})
    @data_class = opts[:data_class] || BayesData
    @corpus = data_class.new('__Corpus__')
    @pools = {'__Corpus__' => @corpus}
    @train_count = 0
    @dirty = true

    # The tokenizer takes an object and returns
    # a list of strings
    @tokenizer = opts[:tokenizer] || Tokenizer.new

    # The combiner combines probabilities
    @combiner = opts[:combiner] || method(:robinson)
  end

  def commit
    save
  end

  # Create a new pool, without actually doing any training.
  def new_pool(pool_name)
    @dirty = true # not always true, but it's simple
    pools[pool_name] ||= data_class.new(pool_name)
  end

  def remove_pool(pool_name)
    pools.delete pool_name
    @dirty = true
  end

  def rename_pool(pool_name, new_name)
    pools[new_name] = pools[pool_name]
    pools[new_name].name = new_name
    remove_pool pool_name
    @dirty = true
  end

  # Merge an existing pool into another.
  # The data from source_pool is merged into dest_pool.
  # The arguments are the names of the pools to be merged.
  # The pool named source_pool is left in tact and you may
  # want to call #remove_pool to get rid of it.
  def merge_pools(dest_pool, source_pool)
    dp = pools[dest_pool]
    pools[source_pool].each do |tok,count|
      if dp[tok]
        dp[tok] += count
      else
        dp[tok] = count
        dp.token_count +=1
      end
    end
    @dirty = true
  end
  
  # Return a list of the (token, count) tuples.
  def pool_data(pool_name)
    pools[pool_name].to_a
  end

  # Return a list of the tokens in this pool.
  def pool_tokens(pool_name)
    pools[pool_name].keys
  end

  def save(file_name='bayesdata.dat')
    File.open(file_name, 'wb') do |f|
      f.write(Marshal.dump(pools))
    end
  end

  def load(file_name='bayesdata.dat')
    File.open(file_name, 'rb') do |f|
      @pools = Marshal.load(f.read)
    end
    @corpus = pools['__Corpus__']
    @dirty = true
  end

  # Return a sorted list of Pool names.
  # Does not include the system pool '__Corpus__'
  def pool_names
    (pools.keys - ['__Corpus__']).sort
  end

  # merges corpora and computes probabilities
  def build_cache
    @cache = {}
    pools.each do |pname, pool|
      next if pname == '__Corpus__'
      pool_count = pool.token_count
      them_count = [@corpus.token_count - pool_count, 1].max
      cache_dict = (@cache[pname] ||= data_class.new(pname))

      @corpus.each do |word, tot_count|
        # for every word in the copus
        # check to see if this pool contains this word
        if (this_count = pool[word].to_f) == 0.to_f
          next
        end
        other_count = tot_count.to_f - this_count
        
        if pool_count.zero?
          good_metric = 1.0
        else
          good_metric = [1.0, other_count / pool_count].min
        end
        bad_metric = [1.0, this_count / them_count].min
        f = bad_metric / (good_metric + bad_metric)

        # PROBABILITY_THRESHOLD
        if (f-0.5).abs >= 0.1
          # GOOD_PROB, BAD_PROB
          cache_dict[word] = [0.0001, [0.9999, f].min].max
        end
      end
    end
  end

  def pool_probs
    if @dirty
      build_cache
      @dirty = false
    end
    @cache
  end
  
  # By default, we expect obj to be a screen and split
  # it on whitespace.
  #
  # Note that this does not change the case.
  # In some applications you may want to lowecase everthing
  # so that "king" and "King" generate the same token.
  #  
  # Override this in your subclass for objects other
  # than text.
  #
  # Alternatively, you can pass in a tokenizer as part of
  # instance creation.
  def get_tokens(obj)
    @tokenizer.tokenize(obj)
  end

  # extracts the probabilities of tokens in a message
  def get_probs(pool, words)
    # This could probably be done better.
    probs = words.inject([]) do |memo, word|
      if pool.has_key? word
        memo << [word, pool[word]]
      end
      memo
    end
    probs = probs.sort_by{ |a| a[1] }
    probs[0, 2048]
  end

  # Train Bayes by telling him that item belongs
  # in pool. uid is optional and may be used to uniquely
  # identify the item that is being trained on.
  def train(pool, item, uid=nil)
    tokens = get_tokens(item)
    pool = (pools[pool] ||= @data_class.new(pool))
    really_train(pool, tokens)
    @corpus.train_count += 1
    pool.train_count += 1
    pool.training << uid if uid
    @dirty = true
  end

  def untrain(pool, item, uid=nil)
    tokens = get_tokens(item)
    return unless pool = pools[pool]
    really_untrain(pool, tokens)
    # I guess we want to count this as additional training?
    @corpus.train_count += 1
    pool.train_count += 1
    pool.training - [uid] if uid
    @dirty = true
  end
  
  private
  def really_train(pool, tokens)
    tokens.each do |token|
      pool[token] = pool[token].to_i + 1
      @corpus[token] = @corpus[token].to_i + 1
    end

    pool.token_count += tokens.size
    @corpus.token_count += tokens.size
  end

  def really_untrain(pool, tokens)
    tokens.each do |token|
      if count = pool[token] && !count.zero?
        if count == 1
          pool.delete token
        else
          pool[token] -= 1
        end
        pool.token_count -= 1
      end
      
      if count = @corpus[token] && !count.zero?
        if count == 1
          @corpus.delete token
        else
          @corpus -= 1
        end
        @corpus.token_count -= 1
      end
    end
  end
  
  public
  def trained_on(msg)
    @cache.values.any?{ |p| p.training.include? msg }
  end

  def guess(msg)
    tokens = Set.new get_tokens(msg)
    pools = pool_probs
    res = {}
    pools.each do |pname,pprobs|
      p = get_probs(pprobs, tokens)
      res[pname] = combiner.call(p, pname) unless p.empty?
    end
    res.to_a.sort_by{ |a| a[1] }
  end

  # computes the probability of a message being spam (Robinson's method)
  # P = 1 - prod(1-p)^(1/n)
  # Q = 1 - prod(p)^(1/n)
  # S = (1 + (P-Q)/(P+Q)) / 2
  # Courtesy of http://christophe.delord.free.fr/en/index.html
  def robinson(probs, ignore)
    nth = 1.0 / probs.size
    p = 1.0 - probs.inject(1.0){ |prod, pr| prod * (1.0-pr[1]) } ** nth
    q = 1.0 - probs.inject(1.0){ |prod, pr| prod * pr[1] } ** nth
    s = (p - q) / (p + q)
    (1 + s) / 2
  end


  # computes the probability of a message being spam (Robinson-Fisher method)
  # H = C-1( -2.ln(prod(p)), 2*n )
  # S = C-1( -2.ln(prod(1-p)), 2*n )
  # I = (1 + H - S) / 2
  # Courtesy of http://christophe.delord.free.fr/en/index.html
  def robinson_fisher(probs, ignore)
    # This is problematic because I'm not sure how I should translate the
    # OverflowError behavior from python
    n = probs.size
    h = chi2P(-2.0 * Math.log(probs.inject(1.0){ |prod, pr| prod * p[1]}), 2*n)
    s = chi2P(-2.0 * Math.log(probs.inject(1.0){ |prod, pr| prod * (1.0 - p[1]) }), 2*n)
    (1 + h - s) / 2
  end

  def inspect
    pools = pool_names.map{ |pn| pools[pn] }
    "#<Bayes: #{pools.inspect}>"
  end

  def size
    @corpus.size
  end
  alias :length :size
end

# A simple regex-based whitespace tokenizer.
# It expects a string and can return all tokens lower-cased
# or in their existing case.
class Tokenizer
  WORD_RE = /\w+/u
  class OddDegreeOfFreedomError; end

  def initialize(lower=false)
    @lower = lower
  end

  def tokenize(obj)
    obj.scan(WORD_RE).each do |word|
      word.downcase! if @lower
      if block_given?
        yield word
      else
        word
      end
    end
  end

  # return P(chisq >= chi, with df degree of freedom)
  # df must be even
  def chi2P(chi, df)
    raise OddDegreeOfFreedomError unless df.even?
    m = chi / 2.0
    sum = Math.exp(-m)
    term = sum.dup
    (1...(df/2)).each do |i|
      term *= m / i
      sum += term
    end
    [sum, 1.0].min
  end
end
end
