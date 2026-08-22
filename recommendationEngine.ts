import {
  Movie,
  User,
  Rating,
  WatchlistItem,
  WatchHistoryItem,
  AIRecommendationResult,
  AIExplanationFactor,
  AgeGroup
} from '../types';

// Extract normalized tokens for content-based vector representation
function extractMovieTokens(movie: Movie): string[] {
  const tokens: string[] = [];

  // Genres (weighted x3)
  movie.genres.forEach(g => {
    tokens.push(`genre:${g.toLowerCase()}`);
    tokens.push(`genre:${g.toLowerCase()}`);
    tokens.push(`genre:${g.toLowerCase()}`);
  });

  // Director (weighted x3)
  const directorParts = movie.director.toLowerCase().split(/[,&]/).map(d => d.trim());
  directorParts.forEach(d => {
    tokens.push(`director:${d}`);
    tokens.push(`director:${d}`);
    tokens.push(`director:${d}`);
  });

  // Cast (weighted x2)
  movie.cast.slice(0, 4).forEach(actor => {
    tokens.push(`actor:${actor.toLowerCase()}`);
    tokens.push(`actor:${actor.toLowerCase()}`);
  });

  // Keywords (weighted x1.5)
  movie.keywords.forEach(kw => {
    tokens.push(`kw:${kw.toLowerCase().trim()}`);
  });

  // Mood tags
  if (movie.moodTags) {
    movie.moodTags.forEach(mood => {
      tokens.push(`mood:${mood.toLowerCase().trim()}`);
    });
  }

  // Language
  tokens.push(`lang:${movie.language.toLowerCase()}`);

  return tokens;
}

// Calculate Cosine Similarity between two token sets
export function calculateCosineSimilarity(tokensA: string[], tokensB: string[]): number {
  if (tokensA.length === 0 || tokensB.length === 0) return 0;

  const freqA = new Map<string, number>();
  const freqB = new Map<string, number>();

  tokensA.forEach(t => freqA.set(t, (freqA.get(t) || 0) + 1));
  tokensB.forEach(t => freqB.set(t, (freqB.get(t) || 0) + 1));

  let dotProduct = 0;
  freqA.forEach((countA, token) => {
    const countB = freqB.get(token) || 0;
    dotProduct += countA * countB;
  });

  let normA = 0;
  freqA.forEach(count => (normA += count * count));
  let normB = 0;
  freqB.forEach(count => (normB += count * count));

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Check if a movie is suitable for a specific age group
export function isMovieSafeForAgeGroup(movie: Movie, ageGroup: AgeGroup): boolean {
  switch (ageGroup) {
    case 'kids':
      if (['16+', '18+', 'R'].includes(movie.ageRating)) return false;
      if (movie.minAge > 12) return false;
      return true;

    case 'teens':
      if (['18+', 'R'].includes(movie.ageRating) && movie.minAge >= 18) return false;
      return true;

    case 'adults':
    case 'seniors':
      return true;

    default:
      return true;
  }
}

// Content Profile description
export function getAgeGroupInfo(ageGroup: AgeGroup): {
  label: string;
  range: string;
  description: string;
  badgeColor: string;
} {
  switch (ageGroup) {
    case 'kids':
      return {
        label: 'Kids',
        range: 'Ages 3–12',
        description: 'Filtered for animation, family adventures, safe comedy, and educational cinema. Adult/18+ content blocked.',
        badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
      };
    case 'teens':
      return {
        label: 'Teens',
        range: 'Ages 13–17',
        description: 'Filtered for teen drama, sci-fi, fantasy, and adventure. Explicit 18+ adult content blocked.',
        badgeColor: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40'
      };
    case 'adults':
      return {
        label: 'Adults (18+)',
        range: 'Ages 18+',
        description: 'Unrestricted access across all cinema genres, psychological thrillers, and R-rated classics.',
        badgeColor: 'bg-rose-500/20 text-rose-400 border-rose-500/40'
      };
    case 'seniors':
      return {
        label: 'Seniors',
        range: 'Ages 60+',
        description: 'Prioritizes timeless classics, rich biographies, historical dramas, romances, and acclaimed masterpieces.',
        badgeColor: 'bg-amber-500/20 text-amber-400 border-amber-500/40'
      };
  }
}

// Build User Taste Profile Vector
export function buildUserTasteTokens(
  user: User,
  movies: Movie[],
  ratings: Rating[],
  watchlist: WatchlistItem[],
  watchHistory: WatchHistoryItem[]
): {
  tokens: string[];
  ratedMovieMap: Map<string, number>;
  genreWeights: Record<string, number>;
} {
  const tokens: string[] = [];
  const movieMap = new Map(movies.map(m => [m.id, m]));
  const ratedMovieMap = new Map<string, number>();
  const genreWeights: Record<string, number> = {};

  // 1. Explicit Preferences
  user.preferences.genres.forEach(g => {
    genreWeights[g] = (genreWeights[g] || 0) + 3;
    for (let i = 0; i < 4; i++) tokens.push(`genre:${g.toLowerCase()}`);
  });

  user.preferences.favoriteDirectors.forEach(d => {
    for (let i = 0; i < 3; i++) tokens.push(`director:${d.toLowerCase()}`);
  });

  user.preferences.favoriteActors.forEach(a => {
    for (let i = 0; i < 2; i++) tokens.push(`actor:${a.toLowerCase()}`);
  });

  user.preferences.languages.forEach(l => {
    tokens.push(`lang:${l.toLowerCase()}`);
  });

  // 2. Star Ratings
  ratings
    .filter(r => r.userId === user.id)
    .forEach(r => {
      ratedMovieMap.set(r.movieId, r.rating);
      const m = movieMap.get(r.movieId);

      if (m) {
        const weight =
          r.rating >= 4 ? r.rating :
          r.rating <= 2 ? -(3 - r.rating) :
          1;

        if (weight > 0) {
          m.genres.forEach(g => {
            genreWeights[g] = (genreWeights[g] || 0) + weight;

            for (let i = 0; i < weight; i++) {
              tokens.push(`genre:${g.toLowerCase()}`);
            }
          });

          tokens.push(`director:${m.director.toLowerCase()}`);
          m.keywords.slice(0, 3).forEach(kw =>
            tokens.push(`kw:${kw.toLowerCase()}`)
          );
        }
      }
    });

  // 3. Watch History
  const userHistory = watchHistory.filter(h => h.userId === user.id);

  userHistory.forEach((h, index) => {
    const m = movieMap.get(h.movieId);

    if (m) {
      const recencyBoost = Math.max(1, 4 - Math.floor(index / 2));

      m.genres.forEach(g => {
        genreWeights[g] = (genreWeights[g] || 0) + 1.5;

        for (let i = 0; i < recencyBoost; i++) {
          tokens.push(`genre:${g.toLowerCase()}`);
        }
      });

      tokens.push(`director:${m.director.toLowerCase()}`);
    }
  });

  // 4. Watchlist
  const userWatchlist = watchlist.filter(w => w.userId === user.id);

  userWatchlist.forEach(w => {
    const m = movieMap.get(w.movieId);

    if (m) {
      m.genres.forEach(g => {
        genreWeights[g] = (genreWeights[g] || 0) + 1;
        tokens.push(`genre:${g.toLowerCase()}`);
      });
    }
  });

  return { tokens, ratedMovieMap, genreWeights };
}

// Generate Explainable AI breakdown
export function generateAIExplanation(
  movie: Movie,
  user: User,
  contentSim: number,
  userPrefScore: number,
  similarWatchedMovie?: Movie
): {
  primaryReason: string;
  factors: AIExplanationFactor[];
  similarToWatched?: string;
} {
  const factors: AIExplanationFactor[] = [];

  // Genre match
  const matchingGenres = movie.genres.filter(g =>
    user.preferences.genres.includes(g)
  );

  if (matchingGenres.length > 0) {
    factors.push({
      title: `${Math.round(userPrefScore * 100)}% Genre Alignment`,
      description: `Matches your favorite genres: ${matchingGenres.join(', ')}`,
      impactScore: Math.min(98, Math.round(userPrefScore * 100)),
      type: 'genre'
    });
  }

  // Director match
  const matchingDirector = user.preferences.favoriteDirectors.find(d =>
    movie.director.toLowerCase().includes(d.toLowerCase())
  );

  if (matchingDirector) {
    factors.push({
      title: 'Director Affinity',
      description: `Directed by ${movie.director}, one of your preferred filmmakers.`,
      impactScore: 92,
      type: 'director'
    });
  }

  // Actor match
  const matchingActors = movie.cast.filter(c =>
    user.preferences.favoriteActors.some(fa =>
      c.toLowerCase().includes(fa.toLowerCase())
    )
  );

  if (matchingActors.length > 0) {
    factors.push({
      title: 'Cast Match',
      description: `Features ${matchingActors.join(', ')}, from your favorite actors list.`,
      impactScore: 88,
      type: 'actor'
    });
  }

  // Similar to watched movie
  if (similarWatchedMovie) {
    factors.push({
      title: `Similar to ${similarWatchedMovie.title}`,
      description: `Shared themes: ${movie.genres
        .filter(g => similarWatchedMovie.genres.includes(g))
        .join(', ')}`,
      impactScore: Math.min(96, Math.round(contentSim * 100)),
      type: 'similarity'
    });
  }

  // Age suitability factor
  factors.push({
    title: `Certified Safe for ${user.ageGroup.toUpperCase()}`,
    description: `Appropriate rating (${movie.ageRating}) verified by CineAI Content Safety.`,
    impactScore: 99,
    type: 'age_suitability'
  });

  // Construct Primary Reason Headline
  let primaryReason = '';

  if (similarWatchedMovie) {
    primaryReason = `Recommended because you watched and enjoyed ${similarWatchedMovie.title}`;
  } else if (matchingGenres.length > 0) {
    primaryReason = `AI found a strong match with your affinity for ${matchingGenres.join(' & ')}`;
  } else if (matchingDirector) {
    primaryReason = `Curated for you based on director ${movie.director}`;
  } else if (
    user.ageGroup === 'seniors' &&
    (
      movie.releaseYear < 2000 ||
      movie.genres.includes('Drama') ||
      movie.genres.includes('Biography')
    )
  ) {
    primaryReason = `Selected for your taste in timeless classics and captivating drama`;
  } else if (user.ageGroup === 'kids') {
    primaryReason = `Top-rated family and animation pick for kids`;
  } else {
    primaryReason = `Top AI recommendation based on your personalized cinema profile`;
  }

  return {
    primaryReason,
    factors: factors.slice(0, 4),
    similarToWatched: similarWatchedMovie?.title
  };
}

// Main Hybrid Recommendation Engine
export function getHybridRecommendations(
  user: User,
  movies: Movie[],
  ratings: Rating[],
  watchlist: WatchlistItem[],
  watchHistory: WatchHistoryItem[],
  limit: number = 20
): AIRecommendationResult[] {
  // 1. Filter out unsafe movies for active age group
  const safeMovies = movies.filter(m =>
    isMovieSafeForAgeGroup(m, user.ageGroup)
  );

  // 2. Build User Taste tokens & rating maps
  const { tokens: userTasteTokens, ratedMovieMap } =
    buildUserTasteTokens(
      user,
      movies,
      ratings,
      watchlist,
      watchHistory
    );

  const userHistory = watchHistory.filter(h => h.userId === user.id);

  let topWatchedMovie: Movie | undefined = undefined;

  if (userHistory.length > 0) {
    topWatchedMovie = movies.find(
      m => m.id === userHistory[0].movieId
    );
  }

  const results: AIRecommendationResult[] = safeMovies.map(movie => {
    const movieTokens = extractMovieTokens(movie);

    // A. Content Similarity Score (0 - 1)
    const contentSimilarity =
      calculateCosineSimilarity(userTasteTokens, movieTokens);

    // B. User Preference Score (0 - 1)
    let genreMatchCount = 0;

    movie.genres.forEach(g => {
      if (user.preferences.genres.includes(g)) {
        genreMatchCount++;
      }
    });

    const genreScore =
      user.preferences.genres.length > 0
        ? genreMatchCount /
          Math.max(1, user.preferences.genres.length)
        : 0.5;

    const directorScore =
      user.preferences.favoriteDirectors.some(d =>
        movie.director.toLowerCase().includes(d.toLowerCase())
      )
        ? 1.0
        : 0.0;

    const actorScore =
      movie.cast.some(c =>
        user.preferences.favoriteActors.some(a =>
          c.toLowerCase().includes(a.toLowerCase())
        )
      )
        ? 1.0
        : 0.0;

    const userPreferenceScore =
      genreScore * 0.6 +
      directorScore * 0.25 +
      actorScore * 0.15;

    // C. Collaborative Filtering Simulation (0 - 1)
    const normalizedRating = Math.min(1.0, movie.rating / 10);
    const popularityScore = Math.min(1.0, movie.popularity / 100);

    const collaborativeScore =
      normalizedRating * 0.7 +
      popularityScore * 0.3;

    // D. Recent Activity / Watch History boost
    let recentActivityScore = 0.5;

    if (topWatchedMovie && movie.id !== topWatchedMovie.id) {
      recentActivityScore =
        calculateCosineSimilarity(
          extractMovieTokens(topWatchedMovie),
          movieTokens
        );
    }

    // E. Age group bonus
    let ageProfileBonus = 0;

    if (
      user.ageGroup === 'seniors' &&
      (
        movie.genres.includes('Drama') ||
        movie.genres.includes('Biography') ||
        movie.releaseYear < 2005
      )
    ) {
      ageProfileBonus = 0.1;
    } else if (
      user.ageGroup === 'kids' &&
      (
        movie.genres.includes('Animation') ||
        movie.genres.includes('Family')
      )
    ) {
      ageProfileBonus = 0.12;
    } else if (
      user.ageGroup === 'teens' &&
      (
        movie.genres.includes('Sci-Fi') ||
        movie.genres.includes('Adventure')
      )
    ) {
      ageProfileBonus = 0.08;
    }

    // F. User Rating modifier
    let ratingModifier = 0;

    const existingRating = ratedMovieMap.get(movie.id);

    if (existingRating !== undefined) {
      if (existingRating <= 2) {
        ratingModifier = -0.4;
      } else if (existingRating >= 4) {
        ratingModifier = 0.05;
      }
    }

    // HYBRID FORMULA
    const rawScore =
      contentSimilarity * 0.40 +
      userPreferenceScore * 0.25 +
      collaborativeScore * 0.20 +
      popularityScore * 0.10 +
      recentActivityScore * 0.05 +
      ageProfileBonus +
      ratingModifier;

    const normalizedAiMatch = Math.min(
      99,
      Math.max(62, Math.round(rawScore * 100))
    );

    const explanation = generateAIExplanation(
      movie,
      user,
      contentSimilarity,
      userPreferenceScore,
      topWatchedMovie && topWatchedMovie.id !== movie.id
        ? topWatchedMovie
        : undefined
    );

    return {
      movie,
      aiMatchScore: normalizedAiMatch,
      breakdown: {
        contentSimilarity: Math.round(contentSimilarity * 100),
        userPreferenceScore: Math.round(userPreferenceScore * 100),
        collaborativeScore: Math.round(collaborativeScore * 100),
        popularityScore: Math.round(popularityScore * 100),
        recentActivityScore: Math.round(recentActivityScore * 100)
      },
      explanation
    };
  });

  return results
    .sort((a, b) => b.aiMatchScore - a.aiMatchScore)
    .slice(0, limit);
}
